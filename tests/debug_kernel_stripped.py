"""Stripped kernel variants to find which section crashes triton-adapter-opt.

All 6 configs of the real kernel crash. Individual patterns (Level 0-7) pass.
Gap: ~60 lines (L7) → ~185 lines (real kernel, APC=off).
This test creates 4 progressively more complete variants of the real kernel.

Run on server:
  rm -rf ~/.triton/cache && python tests/debug_kernel_stripped.py
"""
import torch
import torch_npu  # noqa: F401
from vllm.triton_utils import tl, triton

DEVICE = "npu"


# ============================================================
# V1: Conv1d compute only (~70 lines)
# Section 3 + col0/col1/col2 from prior x tokens (no pool access)
# ============================================================
@triton.jit
def kernel_v1(
    x_ptr, w_ptr, bias_ptr, o_ptr,
    query_start_loc_ptr, batch_ptr, token_chunk_offset_ptr,
    # Dimensions
    dim: tl.constexpr,
    # Strides
    stride_x_dim: tl.constexpr, stride_x_token,
    stride_w_dim: tl.constexpr, stride_w_width: tl.constexpr,
    stride_o_dim: tl.constexpr, stride_o_token,
    # Meta
    pad_slot_id: tl.constexpr,
    HAS_BIAS: tl.constexpr, SILU_ACTIVATION: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """V1: conv1d compute + read col history from x. No pool access."""
    idx_seq = tl.load(batch_ptr + tl.program_id(0)).to(tl.int64)
    chunk_offset = tl.load(token_chunk_offset_ptr + tl.program_id(0))
    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    if USE_PAD_SLOT:
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

    # Read col0/col1/col2 from prior x tokens (or zero if chunk_offset==0)
    if chunk_offset == 0:
        col0 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
        col1 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
        col2 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
    else:
        mask_w = idx_feats < dim
        prior = x_base + (token_offset - 1) * stride_x_token
        col2 = tl.load(prior, mask_w, 0.0)
        col1 = tl.load(prior - 1 * stride_x_token, mask_w, 0.0)
        col0 = tl.load(prior - 2 * stride_x_token, mask_w, 0.0)

    # --- Section 3: Conv1d compute ---
    if HAS_BIAS:
        mask_bias = idx_feats < dim
        acc_preload = tl.load(
            bias_ptr + idx_feats, mask=mask_bias, other=0.0).to(tl.float32)
    else:
        acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)

    x_base_1d = x_base + token_offset * stride_x_token
    mask_w2 = idx_feats < dim
    w_col0 = tl.load(w_base + 0 * stride_w_width, mask_w2, other=0.0)
    w_col1 = tl.load(w_base + 1 * stride_w_width, mask_w2, other=0.0)
    w_col2 = tl.load(w_base + 2 * stride_w_width, mask_w2, other=0.0)
    w_col3 = tl.load(w_base + 3 * stride_w_width, mask_w2, other=0.0)

    mask_x_1d = idx_feats < dim
    for idx_token in range(segment_len):
        acc = acc_preload
        acc += col0 * w_col0
        acc += col1 * w_col1
        acc += col2 * w_col2
        x_ptrs_1d = x_base_1d + idx_token * stride_x_token
        matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
        acc += matrix_x * w_col3
        col0 = col1
        col1 = col2
        col2 = matrix_x
        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))
        mask_1d = (idx_token < segment_len) & (idx_feats < dim)
        o_ptrs = (o_ptr
                  + (sequence_start_index + token_offset + idx_token)
                  * stride_o_token + idx_feats * stride_o_dim)
        tl.store(o_ptrs, acc, mask=mask_1d)


# ============================================================
# V2: V1 + initial state read from pool (1D loads, no 2D store)
# ============================================================
@triton.jit
def kernel_v2(
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
    HAS_BIAS: tl.constexpr, SILU_ACTIVATION: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """V2: V1 + read initial conv state from pool (1D loads). No 2D store."""
    idx_seq = tl.load(batch_ptr + tl.program_id(0)).to(tl.int64)
    chunk_offset = tl.load(token_chunk_offset_ptr + tl.program_id(0))
    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    if USE_PAD_SLOT:
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
        # Read initial state from pool (1D loads per column)
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
    else:
        mask_w = idx_feats < dim
        prior = x_base + (token_offset - 1) * stride_x_token
        col2 = tl.load(prior, mask_w, 0.0)
        col1 = tl.load(prior - stride_x_token, mask_w, 0.0)
        col0 = tl.load(prior - 2 * stride_x_token, mask_w, 0.0)

    # --- Section 3: Conv1d compute (same as V1) ---
    if HAS_BIAS:
        mask_bias = idx_feats < dim
        acc_preload = tl.load(
            bias_ptr + idx_feats, mask=mask_bias, other=0.0).to(tl.float32)
    else:
        acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)

    x_base_1d = x_base + token_offset * stride_x_token
    mask_w2 = idx_feats < dim
    w_col0 = tl.load(w_base + 0 * stride_w_width, mask_w2, other=0.0)
    w_col1 = tl.load(w_base + 1 * stride_w_width, mask_w2, other=0.0)
    w_col2 = tl.load(w_base + 2 * stride_w_width, mask_w2, other=0.0)
    w_col3 = tl.load(w_base + 3 * stride_w_width, mask_w2, other=0.0)

    mask_x_1d = idx_feats < dim
    for idx_token in range(segment_len):
        acc = acc_preload
        acc += col0 * w_col0
        acc += col1 * w_col1
        acc += col2 * w_col2
        x_ptrs_1d = x_base_1d + idx_token * stride_x_token
        matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
        acc += matrix_x * w_col3
        col0 = col1
        col1 = col2
        col2 = matrix_x
        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))
        mask_1d = (idx_token < segment_len) & (idx_feats < dim)
        o_ptrs = (o_ptr
                  + (sequence_start_index + token_offset + idx_token)
                  * stride_o_token + idx_feats * stride_o_dim)
        tl.store(o_ptrs, acc, mask=mask_1d)


# ============================================================
# V3: V2 + final state 2D write (common case: state_len <= seqlen)
# ============================================================
@triton.jit
def kernel_v3(
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
    HAS_BIAS: tl.constexpr, SILU_ACTIVATION: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """V3: V2 + write final conv state to DEST (2D store, common case only)."""
    idx_seq = tl.load(batch_ptr + tl.program_id(0)).to(tl.int64)
    chunk_offset = tl.load(token_chunk_offset_ptr + tl.program_id(0))
    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    if USE_PAD_SLOT:
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

        # ---- NEW in V3: Write final conv_state (common case) ----
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
            loaded_x = tl.load(x_ptrs_2d, mask_x_2d, 0.0)
            idx_tok_c = tl.arange(0, NP2_STATELEN)

            dest_coord = tl.load(
                cache_indices_ptr + idx_seq * stride_cache_indices
            ).to(tl.int64)
            cs_ptrs_target = (
                conv_states_ptr
                + dest_coord * stride_istate_seq
                + (idx_feats * stride_istate_dim)
            )[None, :] + (
                idx_tok_c * stride_istate_token
            )[:, None]
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

    # --- Section 3: Conv1d compute ---
    if HAS_BIAS:
        mask_bias = idx_feats < dim
        acc_preload = tl.load(
            bias_ptr + idx_feats, mask=mask_bias, other=0.0).to(tl.float32)
    else:
        acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)

    x_base_1d = x_base + token_offset * stride_x_token
    mask_w2 = idx_feats < dim
    w_col0 = tl.load(w_base + 0 * stride_w_width, mask_w2, other=0.0)
    w_col1 = tl.load(w_base + 1 * stride_w_width, mask_w2, other=0.0)
    w_col2 = tl.load(w_base + 2 * stride_w_width, mask_w2, other=0.0)
    w_col3 = tl.load(w_base + 3 * stride_w_width, mask_w2, other=0.0)

    mask_x_1d = idx_feats < dim
    for idx_token in range(segment_len):
        acc = acc_preload
        acc += col0 * w_col0
        acc += col1 * w_col1
        acc += col2 * w_col2
        x_ptrs_1d = x_base_1d + idx_token * stride_x_token
        matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
        acc += matrix_x * w_col3
        col0 = col1
        col1 = col2
        col2 = matrix_x
        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))
        mask_1d = (idx_token < segment_len) & (idx_feats < dim)
        o_ptrs = (o_ptr
                  + (sequence_start_index + token_offset + idx_token)
                  * stride_o_token + idx_feats * stride_o_dim)
        tl.store(o_ptrs, acc, mask=mask_1d)


# ============================================================
# V4: V3 + rare case (seqlen < state_len) — adds ~90 lines
# This should be close to the real kernel (IS_APC_ENABLED=False)
# ============================================================
@triton.jit
def kernel_v4(
    x_ptr, w_ptr, bias_ptr,
    conv_states_ptr, cache_indices_ptr, has_initial_states_ptr,
    o_ptr,
    query_start_loc_ptr, batch_ptr, token_chunk_offset_ptr,
    dim: tl.constexpr,
    num_cache_lines,
    stride_x_dim: tl.constexpr, stride_x_token,
    stride_w_dim: tl.constexpr, stride_w_width: tl.constexpr,
    stride_istate_seq, stride_istate_dim: tl.constexpr, stride_istate_token,
    stride_cache_indices,
    stride_o_dim: tl.constexpr, stride_o_token,
    pad_slot_id: tl.constexpr,
    HAS_BIAS: tl.constexpr, SILU_ACTIVATION: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """V4: V3 + rare case (seqlen < state_len). Close to real kernel."""
    idx_seq = tl.load(batch_ptr + tl.program_id(0)).to(tl.int64)
    chunk_offset = tl.load(token_chunk_offset_ptr + tl.program_id(0))
    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    if USE_PAD_SLOT:
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
        conv_state_input_coord = tl.load(
            cache_indices_ptr + idx_seq * stride_cache_indices
        ).to(tl.int64)

        if load_init:
            cs_base = (conv_states_ptr
                       + conv_state_input_coord * stride_istate_seq
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

        # Write final conv_state
        if state_len <= seqlen:
            # Common case
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
            idx_tok_c = tl.arange(0, NP2_STATELEN)
            dest_coord = conv_state_input_coord  # same slot for non-APC
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
            # ---- NEW in V4: Rare case (seqlen < state_len) ----
            if load_init:
                idx_tok_c = tl.arange(0, NP2_STATELEN)
                cs_ptrs_src = (
                    conv_states_ptr
                    + conv_state_input_coord * stride_istate_seq
                    + (idx_feats * stride_istate_dim)[None, :]
                    + ((idx_tok_c + seqlen) * stride_istate_token)[:, None]
                )
                mask_src = (
                    (conv_state_input_coord < num_cache_lines)
                    & ((idx_tok_c + seqlen) < state_len)[:, None]
                    & (idx_feats < dim)[None, :]
                )
                conv_state = tl.load(cs_ptrs_src, mask_src, other=0.0)
                VAL = state_len - seqlen
                x_ptrs_rare = (
                    x_base[None, :]
                    + ((idx_tok_c - VAL) * stride_x_token)[:, None]
                )
                mask_rare = (
                    (idx_tok_c - VAL >= 0)[:, None]
                    & (idx_tok_c - VAL < seqlen)[:, None]
                    & (idx_feats < dim)[None, :]
                )
                loaded_x_rare = tl.load(x_ptrs_rare, mask_rare, 0.0)
                new_state = tl.where(mask_src, conv_state, loaded_x_rare)
                cs_ptrs_dest = (
                    conv_states_ptr
                    + conv_state_input_coord * stride_istate_seq
                    + (idx_feats * stride_istate_dim)
                )[None, :] + (idx_tok_c * stride_istate_token)[:, None]
                mask_dest = (
                    (idx_tok_c < state_len)[:, None]
                    & (idx_feats < dim)[None, :]
                )
                tl.store(cs_ptrs_dest, new_state, mask_dest)
            else:
                idx_tok_c = tl.arange(0, NP2_STATELEN)
                VAL = state_len - seqlen
                x_ptrs_rare2 = (
                    x_base[None, :]
                    + ((idx_tok_c - VAL) * stride_x_token)[:, None]
                )
                mask_rare2 = (
                    (idx_tok_c - VAL >= 0)[:, None]
                    & (idx_tok_c - VAL < seqlen)[:, None]
                    & (idx_feats < dim)[None, :]
                )
                new_state2 = tl.load(x_ptrs_rare2, mask_rare2, 0.0)
                cs_ptrs_dest2 = (
                    conv_states_ptr
                    + conv_state_input_coord * stride_istate_seq
                    + (idx_feats * stride_istate_dim)
                )[None, :] + (idx_tok_c * stride_istate_token)[:, None]
                mask_dest2 = (
                    (idx_tok_c < state_len)[:, None]
                    & (idx_feats < dim)[None, :]
                )
                tl.store(cs_ptrs_dest2, new_state2, mask_dest2)
    else:
        mask_w = idx_feats < dim
        prior = x_base + (token_offset - 1) * stride_x_token
        col2 = tl.load(prior, mask_w, 0.0)
        col1 = tl.load(prior - stride_x_token, mask_w, 0.0)
        col0 = tl.load(prior - 2 * stride_x_token, mask_w, 0.0)

    # --- Section 3: Conv1d compute ---
    if HAS_BIAS:
        mask_bias = idx_feats < dim
        acc_preload = tl.load(
            bias_ptr + idx_feats, mask=mask_bias, other=0.0).to(tl.float32)
    else:
        acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)

    x_base_1d = x_base + token_offset * stride_x_token
    mask_w2 = idx_feats < dim
    w_col0 = tl.load(w_base + 0 * stride_w_width, mask_w2, other=0.0)
    w_col1 = tl.load(w_base + 1 * stride_w_width, mask_w2, other=0.0)
    w_col2 = tl.load(w_base + 2 * stride_w_width, mask_w2, other=0.0)
    w_col3 = tl.load(w_base + 3 * stride_w_width, mask_w2, other=0.0)

    mask_x_1d = idx_feats < dim
    for idx_token in range(segment_len):
        acc = acc_preload
        acc += col0 * w_col0
        acc += col1 * w_col1
        acc += col2 * w_col2
        x_ptrs_1d = x_base_1d + idx_token * stride_x_token
        matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
        acc += matrix_x * w_col3
        col0 = col1
        col1 = col2
        col2 = matrix_x
        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))
        mask_1d = (idx_token < segment_len) & (idx_feats < dim)
        o_ptrs = (o_ptr
                  + (sequence_start_index + token_offset + idx_token)
                  * stride_o_token + idx_feats * stride_o_dim)
        tl.store(o_ptrs, acc, mask=mask_1d)


# ============================================================
# Test runner
# ============================================================
from vllm_ascend.ops.triton.mamba.causal_conv1d import compute_conv1d_grid_npu

DIM = 64
BLOCK_N = 64
BLOCK_M = 8


def make_data():
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


def test_v1():
    (x, w, bias, o, cs, ci, hi, qsl, bp, co, grid, ps) = make_data()
    kernel_v1[grid](
        x, w, bias, o, qsl, bp, co,
        DIM, x.stride(1), x.stride(0),
        w.stride(0), w.stride(1),
        o.stride(1), o.stride(0),
        -1, HAS_BIAS=True, SILU_ACTIVATION=True,
        USE_PAD_SLOT=True, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    torch.npu.synchronize()
    print("✅ V1 PASSED: conv1d compute only (~70 lines)")


def test_v2():
    (x, w, bias, o, cs, ci, hi, qsl, bp, co, grid, ps) = make_data()
    kernel_v2[grid](
        x, w, bias, cs, ci, hi, o, qsl, bp, co,
        DIM, x.stride(1), x.stride(0),
        w.stride(0), w.stride(1),
        cs.stride(0), cs.stride(1), cs.stride(2),
        ci.stride(0), o.stride(1), o.stride(0),
        -1, HAS_BIAS=True, SILU_ACTIVATION=True,
        USE_PAD_SLOT=True, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    torch.npu.synchronize()
    print("✅ V2 PASSED: + initial state read from pool (~110 lines)")


def test_v3():
    (x, w, bias, o, cs, ci, hi, qsl, bp, co, grid, ps) = make_data()
    kernel_v3[grid](
        x, w, bias, cs, ci, hi, o, qsl, bp, co,
        DIM, x.stride(1), x.stride(0),
        w.stride(0), w.stride(1),
        cs.stride(0), cs.stride(1), cs.stride(2),
        ci.stride(0), o.stride(1), o.stride(0),
        -1, HAS_BIAS=True, SILU_ACTIVATION=True,
        USE_PAD_SLOT=True, NP2_STATELEN=4,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    torch.npu.synchronize()
    print("✅ V3 PASSED: + final state 2D write (~160 lines)")


def test_v4():
    (x, w, bias, o, cs, ci, hi, qsl, bp, co, grid, ps) = make_data()
    kernel_v4[grid](
        x, w, bias, cs, ci, hi, o, qsl, bp, co,
        DIM, ps,
        x.stride(1), x.stride(0),
        w.stride(0), w.stride(1),
        cs.stride(0), cs.stride(1), cs.stride(2),
        ci.stride(0), o.stride(1), o.stride(0),
        -1, HAS_BIAS=True, SILU_ACTIVATION=True,
        USE_PAD_SLOT=True, NP2_STATELEN=4,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    torch.npu.synchronize()
    print("✅ V4 PASSED: + rare case (seqlen < state_len) (~250 lines)")


if __name__ == "__main__":
    tests = [
        ("V1: conv1d compute only", test_v1),
        ("V2: + pool state read (1D)", test_v2),
        ("V3: + final state write (2D)", test_v3),
        ("V4: + rare case", test_v4),
    ]
    for label, fn in tests:
        print(f"\n--- {label} ---")
        try:
            fn()
        except Exception as e:
            print(f"❌ {label}: {type(e).__name__}")
            err = str(e).strip().split('\n')
            for line in err[-8:]:
                print(f"   {line}")
