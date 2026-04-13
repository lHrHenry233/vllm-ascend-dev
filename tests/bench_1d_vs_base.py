"""Benchmark: V2 (no state write-back) vs V2g (1D row loop write-back).

Measures the overhead of the 1D row loop state write-back pattern
at realistic Qwen3.5 GDN dimensions.

Run on server:
  rm -rf ~/.triton/cache && python tests/bench_1d_vs_base.py
"""
import time
import torch
import torch_npu  # noqa: F401
from vllm.triton_utils import tl, triton
from vllm_ascend.ops.triton.mamba.causal_conv1d import compute_conv1d_grid_npu

DEVICE = "npu"
BLOCK_M = 8
D_CONV = 4
STATE_LEN = D_CONV - 1  # 3


# ============================================================
# V2_base: Conv1d + pool read, NO state write-back (align-mode)
# ============================================================
@triton.jit
def kernel_base(
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
        # NO state write-back (align-mode baseline)
    else:
        mask_w = idx_feats < dim
        prior = x_base + (token_offset - 1) * stride_x_token
        col2 = tl.load(prior, mask_w, 0.0)
        col1 = tl.load(prior - stride_x_token, mask_w, 0.0)
        col0 = tl.load(prior - 2 * stride_x_token, mask_w, 0.0)

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
# V2g: Conv1d + pool read + 1D ROW LOOP state write-back (all-mode)
# ============================================================
@triton.jit
def kernel_1d_writeback(
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

        # 1D row loop state write-back (all-mode)
        if state_len <= seqlen:
            dest_coord = tl.load(
                cache_indices_ptr + idx_seq * stride_cache_indices
            ).to(tl.int64)
            mask_f2 = idx_feats < dim
            for row in range(state_len):
                src_tok = (seqlen - state_len) + row
                x_row_ptr = (x_ptr
                             + (sequence_start_index + src_tok) * stride_x_token
                             + idx_feats * stride_x_dim)
                row_data = tl.load(x_row_ptr, mask_f2, 0.0)
                cs_row_ptr = (conv_states_ptr
                              + dest_coord * stride_istate_seq
                              + idx_feats * stride_istate_dim
                              + row * stride_istate_token)
                tl.store(cs_row_ptr, row_data, mask_f2)
    else:
        mask_w = idx_feats < dim
        prior = x_base + (token_offset - 1) * stride_x_token
        col2 = tl.load(prior, mask_w, 0.0)
        col1 = tl.load(prior - stride_x_token, mask_w, 0.0)
        col0 = tl.load(prior - 2 * stride_x_token, mask_w, 0.0)

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
# Benchmark harness
# ============================================================
def make_data(batch_size, seqlen, dim, block_n):
    """Create realistic GDN conv1d test data."""
    total_tokens = batch_size * seqlen
    pool_size = batch_size * 4  # enough pool slots

    x = torch.randn(total_tokens, dim, device=DEVICE, dtype=torch.bfloat16)
    w = torch.randn(dim, D_CONV, device=DEVICE, dtype=torch.bfloat16)
    bias = torch.randn(dim, device=DEVICE, dtype=torch.bfloat16)
    o = torch.empty_like(x)

    # conv_states: [pool_size, dim, state_len] (transposed from [pool, state_len, dim])
    cs_raw = torch.randn(pool_size, STATE_LEN, dim, device=DEVICE, dtype=torch.bfloat16)
    conv_states = cs_raw.transpose(1, 2).contiguous()

    # query_start_loc: uniform seqlens
    starts = [i * seqlen for i in range(batch_size + 1)]
    query_start_loc = torch.tensor(starts, dtype=torch.int32, device=DEVICE)

    has_initial = torch.ones(batch_size, dtype=torch.int32, device=DEVICE)

    # cache_indices: [batch, max_blocks] — just sequential pool slots
    max_blocks = 3
    cache_indices = torch.arange(
        batch_size * max_blocks, dtype=torch.int32, device=DEVICE
    ).reshape(batch_size, max_blocks)

    batch_ptr, chunk_offset_ptr, num_programs = compute_conv1d_grid_npu(
        query_start_loc, BLOCK_M, -1, x.device)
    grid = (num_programs, triton.cdiv(dim, block_n))

    return {
        'x': x, 'w': w, 'bias': bias, 'o': o,
        'conv_states': conv_states,
        'cache_indices': cache_indices,
        'has_initial': has_initial,
        'query_start_loc': query_start_loc,
        'batch_ptr': batch_ptr,
        'chunk_offset_ptr': chunk_offset_ptr,
        'grid': grid,
        'dim': dim,
        'block_n': block_n,
    }


def run_kernel(kernel_fn, data, warmup=5, repeat=50):
    """Run a kernel and return average time in microseconds."""
    d = data
    args = (
        d['x'], d['w'], d['bias'],
        d['conv_states'], d['cache_indices'], d['has_initial'],
        d['o'],
        d['query_start_loc'], d['batch_ptr'], d['chunk_offset_ptr'],
        d['dim'],
        d['x'].stride(1), d['x'].stride(0),
        d['w'].stride(0), d['w'].stride(1),
        d['conv_states'].stride(0), d['conv_states'].stride(1), d['conv_states'].stride(2),
        d['cache_indices'].stride(0),
        d['o'].stride(1), d['o'].stride(0),
    )
    kwargs = dict(pad_slot_id=-1, BLOCK_M=BLOCK_M, BLOCK_N=d['block_n'])

    # Warmup (includes compilation)
    for _ in range(warmup):
        kernel_fn[d['grid']](*args, **kwargs)
        torch.npu.synchronize()

    # Timed runs
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        kernel_fn[d['grid']](*args, **kwargs)
    torch.npu.synchronize()
    t1 = time.perf_counter()

    return (t1 - t0) / repeat * 1e6  # microseconds


def main():
    configs = [
        # (batch, seqlen, dim, block_n) — realistic Qwen3.5 GDN params
        (1,   64,   64,  64),   # tiny (debug)
        (1,  128, 1024,  64),   # single seq, small dim
        (4,  256, 1024,  64),   # small batch
        (16, 512, 1024,  64),   # medium batch
        (4,  512, 3840,  64),   # Qwen3.5 actual dim
        (16, 512, 3840,  64),   # Qwen3.5 medium batch
        (4, 2048, 3840,  64),   # Qwen3.5 long seq
    ]

    print("=" * 85)
    print(f"{'Config':>30s} | {'Base (µs)':>10s} | {'1D-WB (µs)':>10s} | {'Overhead':>10s}")
    print("-" * 85)

    for batch, seqlen, dim, block_n in configs:
        label = f"B={batch:2d} L={seqlen:4d} D={dim:4d}"
        data = make_data(batch, seqlen, dim, block_n)

        t_base = run_kernel(kernel_base, data)
        t_1d = run_kernel(kernel_1d_writeback, data)
        overhead = (t_1d - t_base) / t_base * 100

        print(f"{label:>30s} | {t_base:10.1f} | {t_1d:10.1f} | {overhead:+9.1f}%")

    print("=" * 85)
    print("Base = align-mode (no write-back), 1D-WB = all-mode (1D row loop write-back)")
    print("Overhead = extra cost of state write-back as % of total kernel time")


if __name__ == "__main__":
    main()
