"""Correctness test for _causal_conv1d_fwd_kernel_npu with 1D row loop fix.

Verifies:
1. Conv1d output values (forward computation)
2. DEST pool conv_state write-back (common case: seqlen >= 3)
3. DEST pool conv_state write-back (rare case: seqlen < 3, with/without init)
4. APC intermediate block boundary conv_state write-back

Run on server:
  rm -rf ~/.triton/cache && python tests/test_conv1d_correctness.py
"""
import torch
import torch_npu  # noqa: F401
import torch.nn.functional as F

from vllm_ascend.ops.triton.mamba.causal_conv1d import (
    causal_conv1d_fwd_npu,
    compute_conv1d_grid_npu,
)

DEVICE = "npu"
D_CONV = 4
STATE_LEN = D_CONV - 1  # 3


def reference_conv1d(x_seq, weight, bias, initial_state=None):
    """Reference causal conv1d using PyTorch ops.

    Args:
        x_seq: (seqlen, dim) input for one sequence
        weight: (dim, 4) conv weights
        bias: (dim,) bias
        initial_state: (state_len, dim) or None

    Returns:
        output: (seqlen, dim)
        final_state: (state_len, dim) — last 3 tokens of x (or padded)
    """
    seqlen, dim = x_seq.shape
    output = torch.zeros_like(x_seq, dtype=torch.float32)

    # Build history buffer: initial_state (3 tokens) + x_seq
    if initial_state is not None:
        history = torch.cat([initial_state, x_seq], dim=0)  # (state_len + seqlen, dim)
    else:
        history = torch.cat([
            torch.zeros(STATE_LEN, dim, dtype=x_seq.dtype, device=x_seq.device),
            x_seq,
        ], dim=0)

    # Causal conv1d: for each token t, conv = sum(history[t+offset] * weight[offset])
    for t in range(seqlen):
        acc = bias.float() if bias is not None else torch.zeros(dim, device=x_seq.device, dtype=torch.float32)
        for k in range(D_CONV):
            acc = acc + history[t + k].float() * weight[:, k].float()
        # SiLU activation
        acc = acc * torch.sigmoid(acc)
        output[t] = acc

    # Final conv state: last state_len tokens from the full x sequence
    if seqlen >= STATE_LEN:
        final_state = x_seq[-STATE_LEN:]
    else:
        if initial_state is not None:
            # Mix old state with new tokens
            final_state = torch.cat([
                initial_state[seqlen:],
                x_seq
            ], dim=0)
        else:
            final_state = torch.cat([
                torch.zeros(STATE_LEN - seqlen, dim, dtype=x_seq.dtype, device=x_seq.device),
                x_seq
            ], dim=0)

    return output, final_state


def test_common_case():
    """Test 1: Common case — seqlen >= state_len, no APC."""
    print("--- Test 1: Common case (seqlen=16, dim=64) ---")
    B, dim = 2, 64
    seqlens = [16, 8]
    total_tokens = sum(seqlens)

    x = torch.randn(total_tokens, dim, device=DEVICE, dtype=torch.bfloat16)
    weight = torch.randn(dim, D_CONV, device=DEVICE, dtype=torch.bfloat16)
    bias = torch.randn(dim, device=DEVICE, dtype=torch.bfloat16)

    pool_size = 8
    conv_states = torch.zeros(pool_size, dim, STATE_LEN, device=DEVICE, dtype=torch.bfloat16)

    # Initial states in pool slots 0 and 1
    init_state_0 = torch.randn(STATE_LEN, dim, device=DEVICE, dtype=torch.bfloat16)
    init_state_1 = torch.randn(STATE_LEN, dim, device=DEVICE, dtype=torch.bfloat16)
    conv_states[0] = init_state_0.T  # pool layout: (dim, state_len)
    conv_states[1] = init_state_1.T

    query_start_loc = torch.tensor(
        [0, seqlens[0], total_tokens], dtype=torch.int32, device=DEVICE)
    has_initial = torch.tensor([1, 1], dtype=torch.int32, device=DEVICE)
    # cache_indices: [batch, max_blocks] — SOURCE=col0, DEST=col_last
    # For non-APC: source=0, dest=0 (same block)
    cache_indices = torch.tensor(
        [[0, 2, 3], [1, 4, 5]], dtype=torch.int32, device=DEVICE)

    # Run kernel (non-APC mode)
    out = causal_conv1d_fwd_npu(
        x, weight, bias, conv_states, query_start_loc,
        cache_indices=cache_indices,
        has_initial_state=has_initial,
        activation="silu",
        pad_slot_id=-1,
        block_idx_first_scheduled_token=None,
        block_idx_last_scheduled_token=None,
        initial_state_idx=None,
        num_computed_tokens=None,
    )

    # Reference computation
    x0 = x[:seqlens[0]]
    x1 = x[seqlens[0]:]
    ref_out0, ref_state0 = reference_conv1d(x0, weight, bias, init_state_0)
    ref_out1, ref_state1 = reference_conv1d(x1, weight, bias, init_state_1)

    # Check output
    out_f32 = out.float()
    atol, rtol = 0.05, 0.05  # bfloat16 tolerance
    assert torch.allclose(out_f32[:seqlens[0]], ref_out0.to(DEVICE), atol=atol, rtol=rtol), \
        f"Output seq0 mismatch: max_diff={torch.max(torch.abs(out_f32[:seqlens[0]] - ref_out0.to(DEVICE))):.6f}"
    assert torch.allclose(out_f32[seqlens[0]:], ref_out1.to(DEVICE), atol=atol, rtol=rtol), \
        f"Output seq1 mismatch: max_diff={torch.max(torch.abs(out_f32[seqlens[0]:] - ref_out1.to(DEVICE))):.6f}"

    # Check conv state write-back (DEST = cache_indices[seq, current_last_index])
    # In non-APC mode, current_last_index = 0, so DEST = cache_indices[seq, 0]
    dest0 = conv_states[cache_indices[0, 0].item()]  # (dim, state_len)
    dest1 = conv_states[cache_indices[1, 0].item()]

    # Reference final state is (state_len, dim), pool is (dim, state_len)
    ref_dest0 = ref_state0.T.to(DEVICE)  # (dim, state_len)
    ref_dest1 = ref_state1.T.to(DEVICE)

    assert torch.allclose(dest0.float(), ref_dest0.float(), atol=0.01), \
        f"Conv state seq0 mismatch: max_diff={torch.max(torch.abs(dest0.float() - ref_dest0.float())):.6f}"
    assert torch.allclose(dest1.float(), ref_dest1.float(), atol=0.01), \
        f"Conv state seq1 mismatch: max_diff={torch.max(torch.abs(dest1.float() - ref_dest1.float())):.6f}"

    print("✅ Test 1 PASSED: output + conv_state write-back correct")


def test_apc_mode():
    """Test 2: APC mode — block boundary intermediate states."""
    print("\n--- Test 2: APC mode (seqlen=32, block_size=16) ---")
    B, dim = 1, 64
    seqlen = 32
    block_size = 16  # 2 blocks

    x = torch.randn(seqlen, dim, device=DEVICE, dtype=torch.bfloat16)
    weight = torch.randn(dim, D_CONV, device=DEVICE, dtype=torch.bfloat16)
    bias = torch.randn(dim, device=DEVICE, dtype=torch.bfloat16)

    pool_size = 8
    conv_states = torch.zeros(pool_size, dim, STATE_LEN, device=DEVICE, dtype=torch.bfloat16)

    # Initial state in slot 0
    init_state = torch.randn(STATE_LEN, dim, device=DEVICE, dtype=torch.bfloat16)
    conv_states[0] = init_state.T

    query_start_loc = torch.tensor([0, seqlen], dtype=torch.int32, device=DEVICE)
    has_initial = torch.tensor([1], dtype=torch.int32, device=DEVICE)

    # cache_indices: SOURCE=slot0 (initial_state_idx=0), blocks: slot 2, 3
    # block_idx_first_scheduled_token=0 (first block to fill)
    # block_idx_last_scheduled_token=1 (DEST = last block)
    cache_indices = torch.tensor([[0, 2, 3]], dtype=torch.int32, device=DEVICE)
    initial_state_idx = torch.tensor([0], dtype=torch.int32, device=DEVICE)
    block_idx_first = torch.tensor([0], dtype=torch.int32, device=DEVICE)
    block_idx_last = torch.tensor([1], dtype=torch.int32, device=DEVICE)
    num_computed = torch.tensor([0], dtype=torch.int32, device=DEVICE)

    out = causal_conv1d_fwd_npu(
        x, weight, bias, conv_states, query_start_loc,
        cache_indices=cache_indices,
        has_initial_state=has_initial,
        activation="silu",
        pad_slot_id=-1,
        block_idx_first_scheduled_token=block_idx_first,
        block_idx_last_scheduled_token=block_idx_last,
        initial_state_idx=initial_state_idx,
        num_computed_tokens=num_computed,
        block_size_to_align=block_size,
    )

    # Reference output
    ref_out, _ = reference_conv1d(x, weight, bias, init_state)
    out_f32 = out.float()
    atol, rtol = 0.05, 0.05

    assert torch.allclose(out_f32, ref_out.to(DEVICE), atol=atol, rtol=rtol), \
        f"APC output mismatch: max_diff={torch.max(torch.abs(out_f32 - ref_out.to(DEVICE))):.6f}"

    # Check intermediate block boundary state (slot 2 = first block boundary)
    # At block boundary (token 15→16), state should be x[13:16] (last 3 before boundary)
    boundary_state = conv_states[2]  # (dim, state_len)
    expected_boundary = x[block_size - STATE_LEN:block_size].T  # x[13:16].T = (dim, 3)
    assert torch.allclose(boundary_state.float(), expected_boundary.float(), atol=0.01), \
        f"Intermediate state mismatch: max_diff={torch.max(torch.abs(boundary_state.float() - expected_boundary.float())):.6f}"

    # Check final state (slot 3 = DEST = last block)
    final_state = conv_states[3]
    expected_final = x[-STATE_LEN:].T  # x[29:32].T = (dim, 3)
    assert torch.allclose(final_state.float(), expected_final.float(), atol=0.01), \
        f"Final state mismatch: max_diff={torch.max(torch.abs(final_state.float() - expected_final.float())):.6f}"

    print("✅ Test 2 PASSED: APC output + intermediate + final states correct")


def test_rare_case_with_init():
    """Test 3: Rare case — seqlen=2 < state_len=3, with initial state."""
    print("\n--- Test 3: Rare case (seqlen=2, has_init=True) ---")
    B, dim = 1, 64
    seqlen = 2

    x = torch.randn(seqlen, dim, device=DEVICE, dtype=torch.bfloat16)
    weight = torch.randn(dim, D_CONV, device=DEVICE, dtype=torch.bfloat16)
    bias = torch.randn(dim, device=DEVICE, dtype=torch.bfloat16)

    pool_size = 4
    conv_states = torch.zeros(pool_size, dim, STATE_LEN, device=DEVICE, dtype=torch.bfloat16)

    init_state = torch.randn(STATE_LEN, dim, device=DEVICE, dtype=torch.bfloat16)
    conv_states[0] = init_state.T

    query_start_loc = torch.tensor([0, seqlen], dtype=torch.int32, device=DEVICE)
    has_initial = torch.tensor([1], dtype=torch.int32, device=DEVICE)
    cache_indices = torch.tensor([[0, 1, 2]], dtype=torch.int32, device=DEVICE)

    out = causal_conv1d_fwd_npu(
        x, weight, bias, conv_states, query_start_loc,
        cache_indices=cache_indices,
        has_initial_state=has_initial,
        activation="silu",
        pad_slot_id=-1,
    )

    # Reference
    ref_out, ref_state = reference_conv1d(x, weight, bias, init_state)
    out_f32 = out.float()
    atol = 0.05

    assert torch.allclose(out_f32, ref_out.to(DEVICE), atol=atol), \
        f"Rare+init output mismatch: max_diff={torch.max(torch.abs(out_f32 - ref_out.to(DEVICE))):.6f}"

    # Conv state: should be [init_state[2], x[0], x[1]] (shifted by seqlen=2)
    dest_state = conv_states[0]  # (dim, state_len)
    expected = ref_state.T.to(DEVICE)
    assert torch.allclose(dest_state.float(), expected.float(), atol=0.01), \
        f"Rare+init state mismatch: max_diff={torch.max(torch.abs(dest_state.float() - expected.float())):.6f}"

    print("✅ Test 3 PASSED: rare case (seqlen=2, has_init) correct")


def test_rare_case_no_init():
    """Test 4: Rare case — seqlen=1 < state_len=3, no initial state."""
    print("\n--- Test 4: Rare case (seqlen=1, has_init=False) ---")
    B, dim = 1, 64
    seqlen = 1

    x = torch.randn(seqlen, dim, device=DEVICE, dtype=torch.bfloat16)
    weight = torch.randn(dim, D_CONV, device=DEVICE, dtype=torch.bfloat16)
    bias = torch.randn(dim, device=DEVICE, dtype=torch.bfloat16)

    pool_size = 4
    conv_states = torch.zeros(pool_size, dim, STATE_LEN, device=DEVICE, dtype=torch.bfloat16)

    query_start_loc = torch.tensor([0, seqlen], dtype=torch.int32, device=DEVICE)
    has_initial = torch.tensor([0], dtype=torch.int32, device=DEVICE)
    cache_indices = torch.tensor([[0, 1, 2]], dtype=torch.int32, device=DEVICE)

    out = causal_conv1d_fwd_npu(
        x, weight, bias, conv_states, query_start_loc,
        cache_indices=cache_indices,
        has_initial_state=has_initial,
        activation="silu",
        pad_slot_id=-1,
    )

    # Reference
    ref_out, ref_state = reference_conv1d(x, weight, bias, None)
    out_f32 = out.float()
    atol = 0.05

    assert torch.allclose(out_f32, ref_out.to(DEVICE), atol=atol), \
        f"Rare+noinit output mismatch: max_diff={torch.max(torch.abs(out_f32 - ref_out.to(DEVICE))):.6f}"

    # Conv state: [0, 0, x[0]] (zero-padded)
    dest_state = conv_states[0]  # (dim, state_len)
    expected = ref_state.T.to(DEVICE)
    assert torch.allclose(dest_state.float(), expected.float(), atol=0.01), \
        f"Rare+noinit state mismatch: max_diff={torch.max(torch.abs(dest_state.float() - expected.float())):.6f}"

    print("✅ Test 4 PASSED: rare case (seqlen=1, no_init) correct")


def test_mixed_batch():
    """Test 5: Mixed batch with different seqlens."""
    print("\n--- Test 5: Mixed batch (seqlens=[32, 8, 2]) ---")
    dim = 64
    seqlens = [32, 8, 2]
    B = len(seqlens)
    total_tokens = sum(seqlens)

    x = torch.randn(total_tokens, dim, device=DEVICE, dtype=torch.bfloat16)
    weight = torch.randn(dim, D_CONV, device=DEVICE, dtype=torch.bfloat16)
    bias = torch.randn(dim, device=DEVICE, dtype=torch.bfloat16)

    pool_size = 12
    conv_states = torch.zeros(pool_size, dim, STATE_LEN, device=DEVICE, dtype=torch.bfloat16)

    # Init states
    init_states = [
        torch.randn(STATE_LEN, dim, device=DEVICE, dtype=torch.bfloat16)
        for _ in range(B)
    ]
    for i, init in enumerate(init_states):
        conv_states[i] = init.T

    starts = [0]
    for s in seqlens:
        starts.append(starts[-1] + s)
    query_start_loc = torch.tensor(starts, dtype=torch.int32, device=DEVICE)
    has_initial = torch.tensor([1, 1, 1], dtype=torch.int32, device=DEVICE)
    cache_indices = torch.tensor(
        [[0, 3, 4], [1, 5, 6], [2, 7, 8]], dtype=torch.int32, device=DEVICE)

    out = causal_conv1d_fwd_npu(
        x, weight, bias, conv_states, query_start_loc,
        cache_indices=cache_indices,
        has_initial_state=has_initial,
        activation="silu",
        pad_slot_id=-1,
    )

    # Reference per-seq
    atol = 0.05
    offset = 0
    for i, sl in enumerate(seqlens):
        x_seq = x[offset:offset + sl]
        ref_out, ref_state = reference_conv1d(x_seq, weight, bias, init_states[i])
        out_f32 = out[offset:offset + sl].float()
        assert torch.allclose(out_f32, ref_out.to(DEVICE), atol=atol), \
            f"Mixed batch seq{i} output mismatch: max_diff={torch.max(torch.abs(out_f32 - ref_out.to(DEVICE))):.6f}"

        dest_slot = cache_indices[i, 0].item()
        dest_state = conv_states[dest_slot]
        expected = ref_state.T.to(DEVICE)
        assert torch.allclose(dest_state.float(), expected.float(), atol=0.01), \
            f"Mixed batch seq{i} state mismatch: max_diff={torch.max(torch.abs(dest_state.float() - expected.float())):.6f}"
        offset += sl

    print("✅ Test 5 PASSED: mixed batch (seqlens=[32,8,2]) all correct")


def test_larger_dim():
    """Test 6: Qwen3.5 realistic dim."""
    print("\n--- Test 6: Large dim (dim=1024, seqlen=64) ---")
    B, dim, seqlen = 1, 1024, 64

    x = torch.randn(seqlen, dim, device=DEVICE, dtype=torch.bfloat16)
    weight = torch.randn(dim, D_CONV, device=DEVICE, dtype=torch.bfloat16)
    bias = torch.randn(dim, device=DEVICE, dtype=torch.bfloat16)

    pool_size = 4
    conv_states = torch.zeros(pool_size, dim, STATE_LEN, device=DEVICE, dtype=torch.bfloat16)
    init_state = torch.randn(STATE_LEN, dim, device=DEVICE, dtype=torch.bfloat16)
    conv_states[0] = init_state.T

    query_start_loc = torch.tensor([0, seqlen], dtype=torch.int32, device=DEVICE)
    has_initial = torch.tensor([1], dtype=torch.int32, device=DEVICE)
    cache_indices = torch.tensor([[0, 1, 2]], dtype=torch.int32, device=DEVICE)

    out = causal_conv1d_fwd_npu(
        x, weight, bias, conv_states, query_start_loc,
        cache_indices=cache_indices,
        has_initial_state=has_initial,
        activation="silu",
        pad_slot_id=-1,
    )

    ref_out, ref_state = reference_conv1d(x, weight, bias, init_state)
    out_f32 = out.float()

    # Slightly higher tolerance for larger dimensions (more accumulation error)
    atol = 0.1
    max_diff_out = torch.max(torch.abs(out_f32 - ref_out.to(DEVICE))).item()
    assert max_diff_out < atol, \
        f"Large dim output mismatch: max_diff={max_diff_out:.6f}"

    dest_state = conv_states[0]
    expected = ref_state.T.to(DEVICE)
    max_diff_state = torch.max(torch.abs(dest_state.float() - expected.float())).item()
    assert max_diff_state < 0.01, \
        f"Large dim state mismatch: max_diff={max_diff_state:.6f}"

    print(f"✅ Test 6 PASSED: large dim (max_diff_out={max_diff_out:.4f}, state={max_diff_state:.4f})")


if __name__ == "__main__":
    test_common_case()
    test_apc_mode()
    test_rare_case_with_init()
    test_rare_case_no_init()
    test_mixed_batch()
    test_larger_dim()
    print("\n" + "=" * 60)
    print("ALL 6 CORRECTNESS TESTS PASSED ✅")
    print("=" * 60)
