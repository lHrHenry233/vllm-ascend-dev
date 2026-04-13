"""Fine-grained bisect between V2 (passes) and V3 (crashes).

V3 adds: 2D load from x + 2D store to pool, inside chunk_offset==0 branch.
This test isolates exactly which addition causes the crash.

Run on server:
  rm -rf ~/.triton/cache && python tests/debug_v2_to_v3.py
"""
import torch
import torch_npu  # noqa: F401
from vllm.triton_utils import tl, triton
from vllm_ascend.ops.triton.mamba.causal_conv1d import compute_conv1d_grid_npu

DEVICE = "npu"
DIM = 64
BLOCK_N = 64
BLOCK_M = 8


# ============================================================
# V2_base: Same as V2 (passes). Shared foundation.
# ============================================================
def _v2_preamble():
    """Shared code pattern — we inline it in each variant."""
    pass  # Just a marker; actual code is duplicated below.


# ============================================================
# V2a: V2 + 2D LOAD only (no 2D store) inside chunk_offset==0
# ============================================================
@triton.jit
def kernel_v2a(
    x_ptr, w_ptr, bias_ptr,
    conv_states_ptr, cache_indices_ptr, has_initial_states_ptr,
    o_ptr,
    query_start_loc_ptr, batch_ptr, token_chunk_offset_ptr,
    dim: tl.constexpr,
    stride_x_dim: tl.constexpr, stride_x_token,
    stride_w_dim: tl.constexpr, stride_w_width: tl.constexpr,
    stride_istate_seq, stride_istate_dim: tl.constexpr, stride_istate_token,
    stride_cache_indices,
    stride_o_dim: tl.constexpr, stride_o_token,
    pad_slot_id: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """V2 + 2D load from x (no store). Test if 2D load alone crashes."""
    idx_seq = tl.load(batch_ptr + tl.program_id(0)).to(tl.int64)
    chunk_offset = tl.load(token_chunk_offset_ptr + tl.program_id(0))
    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    if idx_seq == pad_slot_id:
        return

    sequence_start_index = tl.load(query_start_loc_ptr + idx_seq)
    sequence_end_index = tl.load(query_start_loc_ptr + idx_seq + 1)
    seqlen = sequence_end_index - sequence_start_index
    token_offset = BLOCK_M * chunk_offset
    segment_len = min(BLOCK_M, seqlen - token_offset)
    x_base = (x_ptr + sequence_start_index * stride_x_token
              + idx_feats * stride_x_dim)
    w_base = w_ptr + idx_feats * stride_w_dim
    state_len = 3

    if chunk_offset == 0:
        load_init = tl.load(has_initial_states_ptr + idx_seq).to(tl.int1)
        if load_init:
            conv_state_coord = tl.load(
                cache_indices_ptr + idx_seq * stride_cache_indices
            ).to(tl.int64)
            cs_base = (conv_states_ptr
                       + conv_state_coord * stride_istate_seq
                       + idx_feats * stride_istate_dim)
            prior = cs_base + (state_len - 1) * stride_istate_token
            mask_f = idx_feats < dim
            col2 = tl.load(prior, mask_f, 0.0)
            col1 = tl.load(prior - stride_istate_token, mask_f, 0.0)
            col0 = tl.load(prior - 2 * stride_istate_token, mask_f, 0.0)
        else:
            col0 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            col1 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            col2 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)

        # ---- NEW: 2D load only (no store) ----
        if state_len <= seqlen:
            idx_tokens_last = (
                (seqlen - state_len) + tl.arange(0, NP2_STATELEN))
            x_ptrs_2d = (
                x_ptr
                + ((sequence_start_index + idx_tokens_last)
                   * stride_x_token)[:, None]
                + (idx_feats * stride_x_dim)[None, :]
            )
            mask_x_2d = (
                (idx_tokens_last >= 0)[:, None]
                & (idx_tokens_last < seqlen)[:, None]
                & (idx_feats < dim)[None, :]
            )
            _loaded = tl.load(x_ptrs_2d, mask_x_2d, 0.0)
            # Just reduce to 1D and discard (no 2D store)
            _sum = tl.sum(_loaded, axis=0)
    else:
        mask_w = idx_feats < dim
        prior = x_base + (token_offset - 1) * stride_x_token
        col2 = tl.load(prior, mask_w, 0.0)
        col1 = tl.load(prior - stride_x_token, mask_w, 0.0)
        col0 = tl.load(prior - 2 * stride_x_token, mask_w, 0.0)

    # --- Conv1d compute ---
    acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)
    x_base_1d = x_base + token_offset * stride_x_token
    mask_w2 = idx_feats < dim
    w0 = tl.load(w_base + 0 * stride_w_width, mask_w2, other=0.0)
    w1 = tl.load(w_base + 1 * stride_w_width, mask_w2, other=0.0)
    w2 = tl.load(w_base + 2 * stride_w_width, mask_w2, other=0.0)
    w3 = tl.load(w_base + 3 * stride_w_width, mask_w2, other=0.0)
    mask_x = idx_feats < dim
    for idx_token in range(segment_len):
        acc = acc_preload + col0 * w0 + col1 * w1 + col2 * w2
        xp = x_base_1d + idx_token * stride_x_token
        mx = tl.load(xp, mask=mask_x)
        acc += mx * w3
        col0 = col1; col1 = col2; col2 = mx
        acc = acc / (1 + tl.exp(-acc))
        m = (idx_token < segment_len) & (idx_feats < dim)
        op = (o_ptr + (sequence_start_index + token_offset + idx_token)
              * stride_o_token + idx_feats * stride_o_dim)
        tl.store(op, acc, mask=m)


# ============================================================
# V2b: V2 + 2D STORE only (store zeros, no 2D load from x)
# ============================================================
@triton.jit
def kernel_v2b(
    x_ptr, w_ptr, bias_ptr,
    conv_states_ptr, cache_indices_ptr, has_initial_states_ptr,
    o_ptr,
    query_start_loc_ptr, batch_ptr, token_chunk_offset_ptr,
    dim: tl.constexpr,
    stride_x_dim: tl.constexpr, stride_x_token,
    stride_w_dim: tl.constexpr, stride_w_width: tl.constexpr,
    stride_istate_seq, stride_istate_dim: tl.constexpr, stride_istate_token,
    stride_cache_indices,
    stride_o_dim: tl.constexpr, stride_o_token,
    pad_slot_id: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """V2 + 2D store to pool (write zeros). Test if 2D store alone crashes."""
    idx_seq = tl.load(batch_ptr + tl.program_id(0)).to(tl.int64)
    chunk_offset = tl.load(token_chunk_offset_ptr + tl.program_id(0))
    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    if idx_seq == pad_slot_id:
        return

    sequence_start_index = tl.load(query_start_loc_ptr + idx_seq)
    sequence_end_index = tl.load(query_start_loc_ptr + idx_seq + 1)
    seqlen = sequence_end_index - sequence_start_index
    token_offset = BLOCK_M * chunk_offset
    segment_len = min(BLOCK_M, seqlen - token_offset)
    x_base = (x_ptr + sequence_start_index * stride_x_token
              + idx_feats * stride_x_dim)
    w_base = w_ptr + idx_feats * stride_w_dim
    state_len = 3

    if chunk_offset == 0:
        load_init = tl.load(has_initial_states_ptr + idx_seq).to(tl.int1)
        if load_init:
            conv_state_coord = tl.load(
                cache_indices_ptr + idx_seq * stride_cache_indices
            ).to(tl.int64)
            cs_base = (conv_states_ptr
                       + conv_state_coord * stride_istate_seq
                       + idx_feats * stride_istate_dim)
            prior = cs_base + (state_len - 1) * stride_istate_token
            mask_f = idx_feats < dim
            col2 = tl.load(prior, mask_f, 0.0)
            col1 = tl.load(prior - stride_istate_token, mask_f, 0.0)
            col0 = tl.load(prior - 2 * stride_istate_token, mask_f, 0.0)
        else:
            col0 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            col1 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            col2 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)

        # ---- NEW: 2D store only (write zeros to pool, no 2D load) ----
        if state_len <= seqlen:
            idx_tok_c = tl.arange(0, NP2_STATELEN)
            dest_coord = tl.load(
                cache_indices_ptr + idx_seq * stride_cache_indices
            ).to(tl.int64)
            cs_ptrs_target = (
                conv_states_ptr
                + dest_coord * stride_istate_seq
                + (idx_feats * stride_istate_dim)
            )[None, :] + (idx_tok_c * stride_istate_token)[:, None]
            mask_cs = (
                (idx_tok_c < state_len)[:, None]
                & (idx_feats < dim)[None, :]
            )
            zeros_2d = tl.zeros((NP2_STATELEN, BLOCK_N),
                                dtype=x_ptr.dtype.element_ty)
            tl.store(cs_ptrs_target, zeros_2d, mask_cs)
    else:
        mask_w = idx_feats < dim
        prior = x_base + (token_offset - 1) * stride_x_token
        col2 = tl.load(prior, mask_w, 0.0)
        col1 = tl.load(prior - stride_x_token, mask_w, 0.0)
        col0 = tl.load(prior - 2 * stride_x_token, mask_w, 0.0)

    # --- Conv1d compute ---
    acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)
    x_base_1d = x_base + token_offset * stride_x_token
    mask_w2 = idx_feats < dim
    w0 = tl.load(w_base + 0 * stride_w_width, mask_w2, other=0.0)
    w1 = tl.load(w_base + 1 * stride_w_width, mask_w2, other=0.0)
    w2 = tl.load(w_base + 2 * stride_w_width, mask_w2, other=0.0)
    w3 = tl.load(w_base + 3 * stride_w_width, mask_w2, other=0.0)
    mask_x = idx_feats < dim
    for idx_token in range(segment_len):
        acc = acc_preload + col0 * w0 + col1 * w1 + col2 * w2
        xp = x_base_1d + idx_token * stride_x_token
        mx = tl.load(xp, mask=mask_x)
        acc += mx * w3
        col0 = col1; col1 = col2; col2 = mx
        acc = acc / (1 + tl.exp(-acc))
        m = (idx_token < segment_len) & (idx_feats < dim)
        op = (o_ptr + (sequence_start_index + token_offset + idx_token)
              * stride_o_token + idx_feats * stride_o_dim)
        tl.store(op, acc, mask=m)


# ============================================================
# V2c: V2 + extra 1D ops to match V3's line count (~160 lines)
# Tests if the crash is purely from IR size, not 2D ops.
# ============================================================
@triton.jit
def kernel_v2c(
    x_ptr, w_ptr, bias_ptr,
    conv_states_ptr, cache_indices_ptr, has_initial_states_ptr,
    o_ptr,
    query_start_loc_ptr, batch_ptr, token_chunk_offset_ptr,
    dim: tl.constexpr,
    stride_x_dim: tl.constexpr, stride_x_token,
    stride_w_dim: tl.constexpr, stride_w_width: tl.constexpr,
    stride_istate_seq, stride_istate_dim: tl.constexpr, stride_istate_token,
    stride_cache_indices,
    stride_o_dim: tl.constexpr, stride_o_token,
    pad_slot_id: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """V2 + extra 1D ops (~160 lines total). Tests if crash is size-related."""
    idx_seq = tl.load(batch_ptr + tl.program_id(0)).to(tl.int64)
    chunk_offset = tl.load(token_chunk_offset_ptr + tl.program_id(0))
    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    if idx_seq == pad_slot_id:
        return

    sequence_start_index = tl.load(query_start_loc_ptr + idx_seq)
    sequence_end_index = tl.load(query_start_loc_ptr + idx_seq + 1)
    seqlen = sequence_end_index - sequence_start_index
    token_offset = BLOCK_M * chunk_offset
    segment_len = min(BLOCK_M, seqlen - token_offset)
    x_base = (x_ptr + sequence_start_index * stride_x_token
              + idx_feats * stride_x_dim)
    w_base = w_ptr + idx_feats * stride_w_dim
    state_len = 3

    if chunk_offset == 0:
        load_init = tl.load(has_initial_states_ptr + idx_seq).to(tl.int1)
        if load_init:
            conv_state_coord = tl.load(
                cache_indices_ptr + idx_seq * stride_cache_indices
            ).to(tl.int64)
            cs_base = (conv_states_ptr
                       + conv_state_coord * stride_istate_seq
                       + idx_feats * stride_istate_dim)
            prior = cs_base + (state_len - 1) * stride_istate_token
            mask_f = idx_feats < dim
            col2 = tl.load(prior, mask_f, 0.0)
            col1 = tl.load(prior - stride_istate_token, mask_f, 0.0)
            col0 = tl.load(prior - 2 * stride_istate_token, mask_f, 0.0)
        else:
            col0 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            col1 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            col2 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)

        # ---- PADDING: Extra 1D ops to reach ~160 lines ----
        # These are dummy 1D loads/stores/arithmetic to test size hypothesis
        if state_len <= seqlen:
            mask_f2 = idx_feats < dim
            # Read some 1D values (wasteful but tests IR size)
            d0 = tl.load(x_base + 0 * stride_x_token, mask_f2, 0.0)
            d1 = tl.load(x_base + 1 * stride_x_token, mask_f2, 0.0)
            d2 = tl.load(x_base + 2 * stride_x_token, mask_f2, 0.0)
            d3 = tl.load(x_base + 3 * stride_x_token, mask_f2, 0.0)
            d4 = tl.load(x_base + 4 * stride_x_token, mask_f2, 0.0)
            # Some arithmetic
            s0 = d0 + d1
            s1 = d2 + d3
            s2 = s0 * s1
            s3 = s2 + d4
            s4 = s3 / (1 + tl.exp(-s3))
            # More 1D loads
            e0 = tl.load(x_base + 5 * stride_x_token, mask_f2, 0.0)
            e1 = tl.load(x_base + 6 * stride_x_token, mask_f2, 0.0)
            e2 = tl.load(x_base + 7 * stride_x_token, mask_f2, 0.0)
            t0 = e0 * e1 + e2
            t1 = t0 + s4
            # Write back to same 1D loc (overwrite x, doesn't matter)
            tl.store(x_base + 0 * stride_x_token, t1, mask_f2)
            tl.store(x_base + 1 * stride_x_token, s4, mask_f2)
    else:
        mask_w = idx_feats < dim
        prior = x_base + (token_offset - 1) * stride_x_token
        col2 = tl.load(prior, mask_w, 0.0)
        col1 = tl.load(prior - stride_x_token, mask_w, 0.0)
        col0 = tl.load(prior - 2 * stride_x_token, mask_w, 0.0)

    # --- Conv1d compute ---
    acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)
    x_base_1d = x_base + token_offset * stride_x_token
    mask_w2 = idx_feats < dim
    w0 = tl.load(w_base + 0 * stride_w_width, mask_w2, other=0.0)
    w1 = tl.load(w_base + 1 * stride_w_width, mask_w2, other=0.0)
    w2 = tl.load(w_base + 2 * stride_w_width, mask_w2, other=0.0)
    w3 = tl.load(w_base + 3 * stride_w_width, mask_w2, other=0.0)
    mask_x = idx_feats < dim
    for idx_token in range(segment_len):
        acc = acc_preload + col0 * w0 + col1 * w1 + col2 * w2
        xp = x_base_1d + idx_token * stride_x_token
        mx = tl.load(xp, mask=mask_x)
        acc += mx * w3
        col0 = col1; col1 = col2; col2 = mx
        acc = acc / (1 + tl.exp(-acc))
        m = (idx_token < segment_len) & (idx_feats < dim)
        op = (o_ptr + (sequence_start_index + token_offset + idx_token)
              * stride_o_token + idx_feats * stride_o_dim)
        tl.store(op, acc, mask=m)


# ============================================================
# Runner
# ============================================================
def make_data():
    B = 2
    total_tokens = 16
    W = 4
    state_len = W - 1
    pool_size = 8
    x = torch.randn(total_tokens, DIM, device=DEVICE, dtype=torch.bfloat16)
    w = torch.randn(DIM, W, device=DEVICE, dtype=torch.bfloat16)
    bias = torch.randn(DIM, device=DEVICE, dtype=torch.bfloat16)
    o = torch.empty_like(x)
    cs_raw = torch.randn(
        pool_size, state_len, DIM, device=DEVICE, dtype=torch.bfloat16)
    conv_states = cs_raw.transpose(1, 2)
    query_start_loc = torch.tensor(
        [0, 8, 16], dtype=torch.int32, device=DEVICE)
    has_initial = torch.tensor([1, 1], dtype=torch.int32, device=DEVICE)
    cache_indices = torch.tensor(
        [[0, 1, 2], [3, 4, 5]], dtype=torch.int32, device=DEVICE)
    batch_ptr, chunk_offset_ptr, num_programs = compute_conv1d_grid_npu(
        query_start_loc, BLOCK_M, -1, x.device)
    grid = (num_programs, triton.cdiv(DIM, BLOCK_N))
    return (x, w, bias, o, conv_states, cache_indices, has_initial,
            query_start_loc, batch_ptr, chunk_offset_ptr, grid, pool_size)


def run_v2a():
    (x, w, bias, o, cs, ci, hi, qsl, bp, co, grid, ps) = make_data()
    kernel_v2a[grid](
        x, w, bias, cs, ci, hi, o, qsl, bp, co,
        DIM, x.stride(1), x.stride(0),
        w.stride(0), w.stride(1),
        cs.stride(0), cs.stride(1), cs.stride(2),
        ci.stride(0), o.stride(1), o.stride(0),
        -1, NP2_STATELEN=4, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    torch.npu.synchronize()
    print("✅ V2a PASSED: V2 + 2D load only (no store)")


def run_v2b():
    (x, w, bias, o, cs, ci, hi, qsl, bp, co, grid, ps) = make_data()
    kernel_v2b[grid](
        x, w, bias, cs, ci, hi, o, qsl, bp, co,
        DIM, x.stride(1), x.stride(0),
        w.stride(0), w.stride(1),
        cs.stride(0), cs.stride(1), cs.stride(2),
        ci.stride(0), o.stride(1), o.stride(0),
        -1, NP2_STATELEN=4, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    torch.npu.synchronize()
    print("✅ V2b PASSED: V2 + 2D store only (write zeros)")


def run_v2c():
    (x, w, bias, o, cs, ci, hi, qsl, bp, co, grid, ps) = make_data()
    kernel_v2c[grid](
        x, w, bias, cs, ci, hi, o, qsl, bp, co,
        DIM, x.stride(1), x.stride(0),
        w.stride(0), w.stride(1),
        cs.stride(0), cs.stride(1), cs.stride(2),
        ci.stride(0), o.stride(1), o.stride(0),
        -1, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    torch.npu.synchronize()
    print("✅ V2c PASSED: V2 + extra 1D ops (same line count as V3)")


# ============================================================
# V2d: V2 + 2D LOAD + 2D STORE (both, but INDEPENDENT data)
# ============================================================
@triton.jit
def kernel_v2d(
    x_ptr, w_ptr, bias_ptr,
    conv_states_ptr, cache_indices_ptr, has_initial_states_ptr,
    o_ptr,
    query_start_loc_ptr, batch_ptr, token_chunk_offset_ptr,
    dim: tl.constexpr,
    stride_x_dim: tl.constexpr, stride_x_token,
    stride_w_dim: tl.constexpr, stride_w_width: tl.constexpr,
    stride_istate_seq, stride_istate_dim: tl.constexpr, stride_istate_token,
    stride_cache_indices,
    stride_o_dim: tl.constexpr, stride_o_token,
    pad_slot_id: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """V2 + 2D load AND 2D store (independent). Tests combination."""
    idx_seq = tl.load(batch_ptr + tl.program_id(0)).to(tl.int64)
    chunk_offset = tl.load(token_chunk_offset_ptr + tl.program_id(0))
    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    if idx_seq == pad_slot_id:
        return

    sequence_start_index = tl.load(query_start_loc_ptr + idx_seq)
    sequence_end_index = tl.load(query_start_loc_ptr + idx_seq + 1)
    seqlen = sequence_end_index - sequence_start_index
    token_offset = BLOCK_M * chunk_offset
    segment_len = min(BLOCK_M, seqlen - token_offset)
    x_base = (x_ptr + sequence_start_index * stride_x_token
              + idx_feats * stride_x_dim)
    w_base = w_ptr + idx_feats * stride_w_dim
    state_len = 3

    if chunk_offset == 0:
        load_init = tl.load(has_initial_states_ptr + idx_seq).to(tl.int1)
        if load_init:
            conv_state_coord = tl.load(
                cache_indices_ptr + idx_seq * stride_cache_indices
            ).to(tl.int64)
            cs_base = (conv_states_ptr
                       + conv_state_coord * stride_istate_seq
                       + idx_feats * stride_istate_dim)
            prior = cs_base + (state_len - 1) * stride_istate_token
            mask_f = idx_feats < dim
            col2 = tl.load(prior, mask_f, 0.0)
            col1 = tl.load(prior - stride_istate_token, mask_f, 0.0)
            col0 = tl.load(prior - 2 * stride_istate_token, mask_f, 0.0)
        else:
            col0 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            col1 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            col2 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)

        if state_len <= seqlen:
            # --- 2D load from x (same as V2a) ---
            idx_tokens_last = (
                (seqlen - state_len) + tl.arange(0, NP2_STATELEN))
            x_ptrs_2d = (
                x_ptr
                + ((sequence_start_index + idx_tokens_last)
                   * stride_x_token)[:, None]
                + (idx_feats * stride_x_dim)[None, :]
            )
            mask_x_2d = (
                (idx_tokens_last >= 0)[:, None]
                & (idx_tokens_last < seqlen)[:, None]
                & (idx_feats < dim)[None, :]
            )
            _loaded = tl.load(x_ptrs_2d, mask_x_2d, 0.0)
            _sum = tl.sum(_loaded, axis=0)  # reduce to 1D, discard

            # --- 2D store to pool (independent, write zeros) ---
            idx_tok_c = tl.arange(0, NP2_STATELEN)
            dest_coord = tl.load(
                cache_indices_ptr + idx_seq * stride_cache_indices
            ).to(tl.int64)
            cs_ptrs_target = (
                conv_states_ptr
                + dest_coord * stride_istate_seq
                + (idx_feats * stride_istate_dim)
            )[None, :] + (idx_tok_c * stride_istate_token)[:, None]
            mask_cs = (
                (idx_tok_c < state_len)[:, None]
                & (idx_feats < dim)[None, :]
            )
            zeros_2d = tl.zeros((NP2_STATELEN, BLOCK_N),
                                dtype=x_ptr.dtype.element_ty)
            tl.store(cs_ptrs_target, zeros_2d, mask_cs)
    else:
        mask_w = idx_feats < dim
        prior = x_base + (token_offset - 1) * stride_x_token
        col2 = tl.load(prior, mask_w, 0.0)
        col1 = tl.load(prior - stride_x_token, mask_w, 0.0)
        col0 = tl.load(prior - 2 * stride_x_token, mask_w, 0.0)

    # --- Conv1d compute ---
    acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)
    x_base_1d = x_base + token_offset * stride_x_token
    mask_w2 = idx_feats < dim
    w0 = tl.load(w_base + 0 * stride_w_width, mask_w2, other=0.0)
    w1 = tl.load(w_base + 1 * stride_w_width, mask_w2, other=0.0)
    w2 = tl.load(w_base + 2 * stride_w_width, mask_w2, other=0.0)
    w3 = tl.load(w_base + 3 * stride_w_width, mask_w2, other=0.0)
    mask_x = idx_feats < dim
    for idx_token in range(segment_len):
        acc = acc_preload + col0 * w0 + col1 * w1 + col2 * w2
        xp = x_base_1d + idx_token * stride_x_token
        mx = tl.load(xp, mask=mask_x)
        acc += mx * w3
        col0 = col1; col1 = col2; col2 = mx
        acc = acc / (1 + tl.exp(-acc))
        m = (idx_token < segment_len) & (idx_feats < dim)
        op = (o_ptr + (sequence_start_index + token_offset + idx_token)
              * stride_o_token + idx_feats * stride_o_dim)
        tl.store(op, acc, mask=m)


# ============================================================
# V2e: V2 + 2D LOAD -> 2D STORE (data flows from load to store)
# This is the ACTUAL V3 pattern from debug_kernel_stripped.py
# ============================================================
@triton.jit
def kernel_v2e(
    x_ptr, w_ptr, bias_ptr,
    conv_states_ptr, cache_indices_ptr, has_initial_states_ptr,
    o_ptr,
    query_start_loc_ptr, batch_ptr, token_chunk_offset_ptr,
    dim: tl.constexpr,
    stride_x_dim: tl.constexpr, stride_x_token,
    stride_w_dim: tl.constexpr, stride_w_width: tl.constexpr,
    stride_istate_seq, stride_istate_dim: tl.constexpr, stride_istate_token,
    stride_cache_indices,
    stride_o_dim: tl.constexpr, stride_o_token,
    pad_slot_id: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """V2 + 2D load -> 2D store (data dependency). Matches real V3."""
    idx_seq = tl.load(batch_ptr + tl.program_id(0)).to(tl.int64)
    chunk_offset = tl.load(token_chunk_offset_ptr + tl.program_id(0))
    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    if idx_seq == pad_slot_id:
        return

    sequence_start_index = tl.load(query_start_loc_ptr + idx_seq)
    sequence_end_index = tl.load(query_start_loc_ptr + idx_seq + 1)
    seqlen = sequence_end_index - sequence_start_index
    token_offset = BLOCK_M * chunk_offset
    segment_len = min(BLOCK_M, seqlen - token_offset)
    x_base = (x_ptr + sequence_start_index * stride_x_token
              + idx_feats * stride_x_dim)
    w_base = w_ptr + idx_feats * stride_w_dim
    state_len = 3

    if chunk_offset == 0:
        load_init = tl.load(has_initial_states_ptr + idx_seq).to(tl.int1)
        if load_init:
            conv_state_coord = tl.load(
                cache_indices_ptr + idx_seq * stride_cache_indices
            ).to(tl.int64)
            cs_base = (conv_states_ptr
                       + conv_state_coord * stride_istate_seq
                       + idx_feats * stride_istate_dim)
            prior = cs_base + (state_len - 1) * stride_istate_token
            mask_f = idx_feats < dim
            col2 = tl.load(prior, mask_f, 0.0)
            col1 = tl.load(prior - stride_istate_token, mask_f, 0.0)
            col0 = tl.load(prior - 2 * stride_istate_token, mask_f, 0.0)
        else:
            col0 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            col1 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
            col2 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)

        if state_len <= seqlen:
            # --- 2D load from x ---
            idx_tokens_last = (
                (seqlen - state_len) + tl.arange(0, NP2_STATELEN))
            x_ptrs_2d = (
                x_ptr
                + ((sequence_start_index + idx_tokens_last)
                   * stride_x_token)[:, None]
                + (idx_feats * stride_x_dim)[None, :]
            )
            mask_x_2d = (
                (idx_tokens_last >= 0)[:, None]
                & (idx_tokens_last < seqlen)[:, None]
                & (idx_feats < dim)[None, :]
            )
            loaded_x = tl.load(x_ptrs_2d, mask_x_2d, 0.0)

            # --- 2D store: write loaded_x to pool (DATA DEPENDENCY) ---
            idx_tok_c = tl.arange(0, NP2_STATELEN)
            dest_coord = tl.load(
                cache_indices_ptr + idx_seq * stride_cache_indices
            ).to(tl.int64)
            cs_ptrs_target = (
                conv_states_ptr
                + dest_coord * stride_istate_seq
                + (idx_feats * stride_istate_dim)
            )[None, :] + (idx_tok_c * stride_istate_token)[:, None]
            mask_cs = (
                (idx_tok_c < state_len)[:, None]
                & (idx_feats < dim)[None, :]
            )
            tl.store(cs_ptrs_target, loaded_x, mask_cs)
    else:
        mask_w = idx_feats < dim
        prior = x_base + (token_offset - 1) * stride_x_token
        col2 = tl.load(prior, mask_w, 0.0)
        col1 = tl.load(prior - stride_x_token, mask_w, 0.0)
        col0 = tl.load(prior - 2 * stride_x_token, mask_w, 0.0)

    # --- Conv1d compute ---
    acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)
    x_base_1d = x_base + token_offset * stride_x_token
    mask_w2 = idx_feats < dim
    w0 = tl.load(w_base + 0 * stride_w_width, mask_w2, other=0.0)
    w1 = tl.load(w_base + 1 * stride_w_width, mask_w2, other=0.0)
    w2 = tl.load(w_base + 2 * stride_w_width, mask_w2, other=0.0)
    w3 = tl.load(w_base + 3 * stride_w_width, mask_w2, other=0.0)
    mask_x = idx_feats < dim
    for idx_token in range(segment_len):
        acc = acc_preload + col0 * w0 + col1 * w1 + col2 * w2
        xp = x_base_1d + idx_token * stride_x_token
        mx = tl.load(xp, mask=mask_x)
        acc += mx * w3
        col0 = col1; col1 = col2; col2 = mx
        acc = acc / (1 + tl.exp(-acc))
        m = (idx_token < segment_len) & (idx_feats < dim)
        op = (o_ptr + (sequence_start_index + token_offset + idx_token)
              * stride_o_token + idx_feats * stride_o_dim)
        tl.store(op, acc, mask=m)


def run_v2d():
    (x, w, bias, o, cs, ci, hi, qsl, bp, co, grid, ps) = make_data()
    kernel_v2d[grid](
        x, w, bias, cs, ci, hi, o, qsl, bp, co,
        DIM, x.stride(1), x.stride(0),
        w.stride(0), w.stride(1),
        cs.stride(0), cs.stride(1), cs.stride(2),
        ci.stride(0), o.stride(1), o.stride(0),
        -1, NP2_STATELEN=4, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    torch.npu.synchronize()
    print("✅ V2d PASSED: V2 + 2D load + 2D store (independent)")


def run_v2e():
    (x, w, bias, o, cs, ci, hi, qsl, bp, co, grid, ps) = make_data()
    kernel_v2e[grid](
        x, w, bias, cs, ci, hi, o, qsl, bp, co,
        DIM, x.stride(1), x.stride(0),
        w.stride(0), w.stride(1),
        cs.stride(0), cs.stride(1), cs.stride(2),
        ci.stride(0), o.stride(1), o.stride(0),
        -1, NP2_STATELEN=4, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    torch.npu.synchronize()
    print("✅ V2e PASSED: V2 + 2D load -> 2D store (data dep)")


if __name__ == "__main__":
    tests = [
        ("V2a: + 2D load only (no store)", run_v2a),
        ("V2b: + 2D store only (write zeros)", run_v2b),
        ("V2c: + extra 1D ops (size test)", run_v2c),
        ("V2d: + 2D load + 2D store (independent)", run_v2d),
        ("V2e: + 2D load -> 2D store (data dep, real V3)", run_v2e),
    ]
    for label, fn in tests:
        print(f"\n--- {label} ---")
        try:
            fn()
        except Exception as e:
            print(f"❌ {label}: {type(e).__name__}")
            err = str(e).strip().split('\n')
            for line in err[-6:]:
                print(f"   {line}")
