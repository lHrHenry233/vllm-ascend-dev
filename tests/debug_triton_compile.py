"""Minimal Triton kernel tests to isolate NPU compiler SIGSEGV.

Run on server:
  python tests/debug_triton_compile.py

Each level adds complexity. Find the first level that crashes.
"""
import torch
import torch_npu  # noqa: F401
from vllm.triton_utils import tl, triton

DEVICE = "npu"
DIM = 64
BLOCK_N = 64  # smaller than production 256
BLOCK_M = 8


# ============================================================
# Level 0: Bare minimum — 1D load + store, no branching
# ============================================================
@triton.jit
def kernel_level0(
    x_ptr, o_ptr,
    dim: tl.constexpr,
    stride_x_token,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = idx < dim
    x = tl.load(x_ptr + pid * stride_x_token + idx, mask=mask, other=0.0)
    tl.store(o_ptr + pid * stride_x_token + idx, x, mask=mask)


def test_level0():
    x = torch.randn(4, DIM, device=DEVICE, dtype=torch.bfloat16)
    o = torch.empty_like(x)
    grid = (4, triton.cdiv(DIM, BLOCK_N))
    kernel_level0[grid](x, o, DIM, x.stride(0), BLOCK_N=BLOCK_N)
    assert torch.allclose(x, o), "Level 0 output mismatch"
    print("✅ Level 0 PASSED: bare 1D load/store")


# ============================================================
# Level 1: Add a for loop (dynamic range via tl.minimum)
# ============================================================
@triton.jit
def kernel_level1(
    x_ptr, w_ptr, o_ptr,
    dim: tl.constexpr,
    stride_x_token,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = idx < dim

    w0 = tl.load(w_ptr + idx * 4 + 0, mask=mask, other=0.0)
    w1 = tl.load(w_ptr + idx * 4 + 1, mask=mask, other=0.0)
    w2 = tl.load(w_ptr + idx * 4 + 2, mask=mask, other=0.0)
    w3 = tl.load(w_ptr + idx * 4 + 3, mask=mask, other=0.0)

    col0 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
    col1 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
    col2 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)

    base = x_ptr + pid * BLOCK_M * stride_x_token
    for i in range(BLOCK_M):
        acc = col0 * w0 + col1 * w1 + col2 * w2
        xv = tl.load(base + i * stride_x_token + idx, mask=mask, other=0.0)
        acc += xv * w3
        col0 = col1
        col1 = col2
        col2 = xv
        tl.store(o_ptr + (pid * BLOCK_M + i) * stride_x_token + idx, acc, mask=mask)


def test_level1():
    seq = 32
    x = torch.randn(seq, DIM, device=DEVICE, dtype=torch.bfloat16)
    w = torch.randn(DIM, 4, device=DEVICE, dtype=torch.bfloat16)
    o = torch.empty_like(x)
    grid = (seq // BLOCK_M, triton.cdiv(DIM, BLOCK_N))
    kernel_level1[grid](x, w, o, DIM, x.stride(0), BLOCK_N=BLOCK_N, BLOCK_M=BLOCK_M)
    print("✅ Level 1 PASSED: for loop + conv1d compute")


# ============================================================
# Level 2: Add simple if branch (constexpr)
# ============================================================
@triton.jit
def kernel_level2(
    x_ptr, w_ptr, bias_ptr, o_ptr,
    dim: tl.constexpr,
    stride_x_token,
    HAS_BIAS: tl.constexpr,
    SILU: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = idx < dim

    w0 = tl.load(w_ptr + idx * 4 + 0, mask=mask, other=0.0)
    w1 = tl.load(w_ptr + idx * 4 + 1, mask=mask, other=0.0)
    w2 = tl.load(w_ptr + idx * 4 + 2, mask=mask, other=0.0)
    w3 = tl.load(w_ptr + idx * 4 + 3, mask=mask, other=0.0)

    if HAS_BIAS:
        bias = tl.load(bias_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    else:
        bias = tl.zeros((BLOCK_N,), dtype=tl.float32)

    col0 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
    col1 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
    col2 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)

    base = x_ptr + pid * BLOCK_M * stride_x_token
    for i in range(BLOCK_M):
        acc = bias + col0 * w0 + col1 * w1 + col2 * w2
        xv = tl.load(base + i * stride_x_token + idx, mask=mask, other=0.0)
        acc += xv * w3
        if SILU:
            acc = acc / (1 + tl.exp(-acc))
        col0 = col1
        col1 = col2
        col2 = xv
        tl.store(o_ptr + (pid * BLOCK_M + i) * stride_x_token + idx, acc, mask=mask)


def test_level2():
    seq = 32
    x = torch.randn(seq, DIM, device=DEVICE, dtype=torch.bfloat16)
    w = torch.randn(DIM, 4, device=DEVICE, dtype=torch.bfloat16)
    b = torch.randn(DIM, device=DEVICE, dtype=torch.bfloat16)
    o = torch.empty_like(x)
    grid = (seq // BLOCK_M, triton.cdiv(DIM, BLOCK_N))
    kernel_level2[grid](x, w, b, o, DIM, x.stride(0),
                        HAS_BIAS=True, SILU=True, BLOCK_N=BLOCK_N, BLOCK_M=BLOCK_M)
    print("✅ Level 2 PASSED: constexpr if + silu activation")


# ============================================================
# Level 3: Add runtime if branch (data-dependent)
# ============================================================
@triton.jit
def kernel_level3(
    x_ptr, w_ptr, o_ptr,
    batch_ptr, query_start_loc_ptr,
    dim: tl.constexpr,
    stride_x_token,
    pad_slot_id: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_seq = tl.load(batch_ptr + tl.program_id(0)).to(tl.int64)
    idx = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = idx < dim

    # Runtime branch: skip padded entries
    if pid_seq == pad_slot_id:
        return

    seq_start = tl.load(query_start_loc_ptr + pid_seq)
    seq_end = tl.load(query_start_loc_ptr + pid_seq + 1)
    seqlen = seq_end - seq_start

    w0 = tl.load(w_ptr + idx * 4 + 0, mask=mask, other=0.0)
    w1 = tl.load(w_ptr + idx * 4 + 1, mask=mask, other=0.0)
    w2 = tl.load(w_ptr + idx * 4 + 2, mask=mask, other=0.0)
    w3 = tl.load(w_ptr + idx * 4 + 3, mask=mask, other=0.0)

    col0 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
    col1 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
    col2 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)

    base = x_ptr + seq_start * stride_x_token
    seg_len = tl.minimum(BLOCK_M, seqlen)
    for i in range(BLOCK_M):
        if i < seg_len:
            acc = col0 * w0 + col1 * w1 + col2 * w2
            xv = tl.load(base + i * stride_x_token + idx, mask=mask, other=0.0)
            acc += xv * w3
            col0 = col1
            col1 = col2
            col2 = xv
            tl.store(o_ptr + (seq_start + i) * stride_x_token + idx, acc, mask=mask)


def test_level3():
    seqlens = [16, 8, 8]
    total = sum(seqlens)
    x = torch.randn(total, DIM, device=DEVICE, dtype=torch.bfloat16)
    w = torch.randn(DIM, 4, device=DEVICE, dtype=torch.bfloat16)
    o = torch.empty_like(x)

    qsl = torch.tensor([0] + list(torch.cumsum(torch.tensor(seqlens), 0).numpy()),
                        dtype=torch.int32, device=DEVICE)
    # One program per seq (simplified: 1 chunk per seq since seqlen <= BLOCK_M*2)
    batch_ptr = torch.tensor([0, 0, 1, 2], dtype=torch.int32, device=DEVICE)
    grid = (len(batch_ptr), triton.cdiv(DIM, BLOCK_N))
    kernel_level3[grid](x, w, o, batch_ptr, qsl, DIM, x.stride(0),
                        pad_slot_id=-1, BLOCK_N=BLOCK_N, BLOCK_M=BLOCK_M)
    print("✅ Level 3 PASSED: runtime if + early return + dynamic seqlen")


# ============================================================
# Level 4: Add 2D masked load/store (state read/write)
# ============================================================
@triton.jit
def kernel_level4(
    x_ptr, conv_states_ptr, o_ptr,
    dim: tl.constexpr,
    stride_x_token,
    stride_cs_seq, stride_cs_tok,
    NP2_SL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Test 2D load/store pattern: conv_states[slot, :, :state_len]."""
    pid = tl.program_id(0)  # slot index
    idx = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_f = idx < dim

    state_len = 3
    idx_tok = tl.arange(0, NP2_SL)

    # 2D load: conv_states[pid, idx_feats, idx_tok]
    ptrs = (
        conv_states_ptr
        + pid * stride_cs_seq
        + idx[None, :]  # (1, BLOCK_N) — dim axis
        + (idx_tok * stride_cs_tok)[:, None]  # (NP2_SL, 1) — tok axis
    )
    mask_2d = (idx_tok < state_len)[:, None] & mask_f[None, :]
    state = tl.load(ptrs, mask=mask_2d, other=0.0)

    # 2D store back (identity)
    tl.store(ptrs, state, mask=mask_2d)

    # 1D output: reduce 2D state to 1D via sum over token dim
    reduced = tl.sum(state, axis=0)  # (BLOCK_N,)
    tl.store(o_ptr + pid * stride_x_token + idx, reduced, mask=mask_f)


def test_level4():
    N = 4
    state_len = 3
    cs = torch.randn(N, DIM, state_len, device=DEVICE, dtype=torch.bfloat16)
    o = torch.empty(N, DIM, device=DEVICE, dtype=torch.bfloat16)
    grid = (N, triton.cdiv(DIM, BLOCK_N))
    kernel_level4[grid](
        torch.empty(1, device=DEVICE), cs, o,
        DIM, DIM, cs.stride(0), cs.stride(2),
        NP2_SL=4, BLOCK_N=BLOCK_N)
    print("✅ Level 4 PASSED: 2D masked load/store (state pattern)")


# ============================================================
# Level 5: Add indirect addressing (load index, then use as ptr)
# ============================================================
@triton.jit
def kernel_level5(
    x_ptr, conv_states_ptr, cache_indices_ptr, o_ptr,
    dim: tl.constexpr,
    stride_x_token,
    stride_cs_seq, stride_cs_tok,
    NP2_SL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Test indirect addressing: load slot from cache_indices, then index pool."""
    pid = tl.program_id(0)
    idx = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_f = idx < dim

    # Indirect: load pool slot ID from cache_indices
    slot = tl.load(cache_indices_ptr + pid).to(tl.int64)

    state_len = 3
    idx_tok = tl.arange(0, NP2_SL)

    # 2D load from indirectly-addressed slot
    ptrs = (
        conv_states_ptr
        + slot * stride_cs_seq
        + idx[None, :]
        + (idx_tok * stride_cs_tok)[:, None]
    )
    mask_2d = (idx_tok < state_len)[:, None] & mask_f[None, :]
    state = tl.load(ptrs, mask=mask_2d, other=0.0)

    reduced = tl.sum(state, axis=0)  # (BLOCK_N,)
    tl.store(o_ptr + pid * stride_x_token + idx, reduced, mask=mask_f)


def test_level5():
    N = 4
    pool_size = 8
    state_len = 3
    cs = torch.randn(pool_size, DIM, state_len, device=DEVICE, dtype=torch.bfloat16)
    cache_idx = torch.tensor([2, 5, 0, 7], dtype=torch.int32, device=DEVICE)
    o = torch.empty(N, DIM, device=DEVICE, dtype=torch.bfloat16)
    grid = (N, triton.cdiv(DIM, BLOCK_N))
    kernel_level5[grid](
        torch.empty(1, device=DEVICE), cs, cache_idx, o,
        DIM, DIM, cs.stride(0), cs.stride(2),
        NP2_SL=4, BLOCK_N=BLOCK_N)
    print("✅ Level 5 PASSED: indirect addressing (load slot → index pool)")


# ============================================================
# Level 6: Combine conv1d compute + state read + runtime branch
#          (approximates the real kernel without all the nesting)
# ============================================================
@triton.jit
def kernel_level6(
    x_ptr, w_ptr, conv_states_ptr, cache_indices_ptr,
    has_init_ptr, query_start_loc_ptr, batch_ptr, o_ptr,
    dim: tl.constexpr,
    stride_x_token,
    stride_cs_seq, stride_cs_tok,
    pad_slot_id: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_seq = tl.load(batch_ptr + tl.program_id(0)).to(tl.int64)
    chunk_offset = tl.program_id(0)  # simplified: 1 chunk per seq
    idx = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = idx < dim

    if pid_seq == pad_slot_id:
        return

    seq_start = tl.load(query_start_loc_ptr + pid_seq)
    seq_end = tl.load(query_start_loc_ptr + pid_seq + 1)
    seqlen = seq_end - seq_start

    w0 = tl.load(w_ptr + idx * 4 + 0, mask=mask, other=0.0)
    w1 = tl.load(w_ptr + idx * 4 + 1, mask=mask, other=0.0)
    w2 = tl.load(w_ptr + idx * 4 + 2, mask=mask, other=0.0)
    w3 = tl.load(w_ptr + idx * 4 + 3, mask=mask, other=0.0)

    # Read initial state from pool (indirect addressing)
    has_init = tl.load(has_init_ptr + pid_seq).to(tl.int1)
    if has_init:
        slot = tl.load(cache_indices_ptr + pid_seq).to(tl.int64)
        col0 = tl.load(conv_states_ptr + slot * stride_cs_seq + idx + 0 * stride_cs_tok,
                        mask=mask, other=0.0)
        col1 = tl.load(conv_states_ptr + slot * stride_cs_seq + idx + 1 * stride_cs_tok,
                        mask=mask, other=0.0)
        col2 = tl.load(conv_states_ptr + slot * stride_cs_seq + idx + 2 * stride_cs_tok,
                        mask=mask, other=0.0)
    else:
        col0 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
        col1 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)
        col2 = tl.zeros((BLOCK_N,), dtype=x_ptr.dtype.element_ty)

    # Conv1d compute
    base = x_ptr + seq_start * stride_x_token
    seg_len = tl.minimum(BLOCK_M, seqlen)
    for i in range(BLOCK_M):
        if i < seg_len:
            acc = col0 * w0 + col1 * w1 + col2 * w2
            xv = tl.load(base + i * stride_x_token + idx, mask=mask, other=0.0)
            acc = acc + xv * w3
            acc = acc / (1 + tl.exp(-acc))  # silu
            col0 = col1
            col1 = col2
            col2 = xv
            tl.store(o_ptr + (seq_start + i) * stride_x_token + idx, acc, mask=mask)


def test_level6():
    seqlens = [8, 8]
    total = sum(seqlens)
    x = torch.randn(total, DIM, device=DEVICE, dtype=torch.bfloat16)
    w = torch.randn(DIM, 4, device=DEVICE, dtype=torch.bfloat16)
    pool_size = 4
    cs = torch.randn(pool_size, DIM, 3, device=DEVICE, dtype=torch.bfloat16)
    cache_idx = torch.tensor([1, 3], dtype=torch.int32, device=DEVICE)
    has_init = torch.tensor([True, False], dtype=torch.bool, device=DEVICE)
    qsl = torch.tensor([0, 8, 16], dtype=torch.int32, device=DEVICE)
    batch_ptr = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)
    o = torch.empty_like(x)
    grid = (2, triton.cdiv(DIM, BLOCK_N))
    kernel_level6[grid](
        x, w, cs, cache_idx, has_init, qsl, batch_ptr, o,
        DIM, x.stride(0), cs.stride(0), cs.stride(2),
        pad_slot_id=-1, BLOCK_N=BLOCK_N, BLOCK_M=BLOCK_M)
    print("✅ Level 6 PASSED: conv1d + state read + runtime branch")


# ============================================================
# Level 7: Add 2D store for final state write-back
# ============================================================
@triton.jit
def kernel_level7(
    x_ptr, conv_states_ptr, cache_indices_ptr,
    query_start_loc_ptr, batch_ptr, o_ptr,
    dim: tl.constexpr,
    seqlen_val,
    stride_x_token,
    stride_cs_seq, stride_cs_tok,
    NP2_SL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Test: read x[last 3 tokens] → 2D store to conv_states[dest_slot]."""
    pid = tl.program_id(0)
    idx = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_f = idx < dim

    seq_start = tl.load(query_start_loc_ptr + pid)
    seq_end = tl.load(query_start_loc_ptr + pid + 1)
    seqlen = seq_end - seq_start
    state_len = 3

    # 2D load: last state_len tokens from x
    idx_tok = tl.arange(0, NP2_SL)
    x_ptrs = (
        x_ptr
        + ((seq_start + seqlen - state_len + idx_tok) * stride_x_token)[:, None]
        + (idx * 1)[None, :]
    )
    mask_2d = (idx_tok < state_len)[:, None] & mask_f[None, :]
    loaded_x = tl.load(x_ptrs, mask_2d, other=0.0)

    # Indirect: get dest slot
    dest_slot = tl.load(cache_indices_ptr + pid).to(tl.int64)

    # 2D store to conv_states[dest_slot]
    cs_ptrs = (
        conv_states_ptr
        + dest_slot * stride_cs_seq
        + idx[None, :]
        + (idx_tok * stride_cs_tok)[:, None]
    )
    tl.store(cs_ptrs, loaded_x, mask=mask_2d)

    # 1D output to verify
    reduced = tl.sum(loaded_x, axis=0)  # (BLOCK_N,)
    tl.store(o_ptr + pid * dim + idx, reduced, mask=mask_f)


def test_level7():
    seqlens = [16, 8]
    total = sum(seqlens)
    x = torch.randn(total, DIM, device=DEVICE, dtype=torch.bfloat16)
    pool_size = 4
    cs = torch.zeros(pool_size, DIM, 3, device=DEVICE, dtype=torch.bfloat16)
    cache_idx = torch.tensor([1, 3], dtype=torch.int32, device=DEVICE)
    qsl = torch.tensor([0, 16, 24], dtype=torch.int32, device=DEVICE)
    o = torch.empty(2, DIM, device=DEVICE, dtype=torch.bfloat16)
    grid = (2, triton.cdiv(DIM, BLOCK_N))
    kernel_level7[grid](
        x, cs, cache_idx, qsl, torch.empty(1, device=DEVICE), o,
        DIM, 0, x.stride(0), cs.stride(0), cs.stride(2),
        NP2_SL=4, BLOCK_N=BLOCK_N)
    print("✅ Level 7 PASSED: 2D store for final state write-back")


# ============================================================
# Main: run all levels
# ============================================================
if __name__ == "__main__":
    tests = [
        ("Level 0", test_level0),
        ("Level 1", test_level1),
        ("Level 2", test_level2),
        ("Level 3", test_level3),
        ("Level 4", test_level4),
        ("Level 5", test_level5),
        ("Level 6", test_level6),
        ("Level 7", test_level7),
    ]
    for name, fn in tests:
        try:
            fn()
        except Exception as e:
            print(f"❌ {name} FAILED: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            break
    else:
        print("\n🎉 All levels passed! The compiler can handle all patterns.")
