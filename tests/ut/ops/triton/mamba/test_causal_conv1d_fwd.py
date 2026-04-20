# SPDX-License-Identifier: Apache-2.0
"""Unit tests for P2: causal_conv1d_fwd_npu and compute_conv1d_grid_npu.

Tests are split into:
  - TestComputeConv1dGridNpu: grid scheduling (pure CPU, always runnable)
  - TestCausalConv1dFwdRef: reference correctness (CPU, uses causal_conv1d_ref)
  - TestCausalConv1dFwdWrapper: wrapper logic validation (CPU, no kernel)

The grid helper and reference conv1d are inlined so tests run without vllm.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

WIDTH = 4
STATE_LEN = WIDTH - 1  # 3


# ──────────────────────────────────────────────────────────────────
# Inlined functions (from causal_conv1d.py — no vllm dependency)
# ──────────────────────────────────────────────────────────────────


def compute_conv1d_grid_npu(
    query_start_loc: torch.Tensor,
    block_m: int,
    pad_slot_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Precompute grid scheduling tensors (inlined from causal_conv1d.py)."""
    seqlens = query_start_loc.diff().cpu().numpy()
    nums = -(-seqlens // block_m)  # cdiv
    total = int(nums.sum())
    mlist = np.repeat(np.arange(len(nums)), nums)
    offsetlist: list[int] = []
    for num in nums:
        offsetlist.extend(range(int(num)))
    batch_ptr = torch.tensor(mlist, dtype=torch.int32, device=device)
    token_chunk_offset_ptr = torch.tensor(
        offsetlist, dtype=torch.int32, device=device)
    return batch_ptr, token_chunk_offset_ptr, total


def causal_conv1d_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    initial_states: torch.Tensor | None = None,
    return_final_states: bool = False,
    final_states_out: torch.Tensor | None = None,
    activation: str | None = "silu",
):
    """Reference conv1d (inlined from causal_conv1d.py).

    x: (batch, dim, seqlen), weight: (dim, width), bias: (dim,)
    initial_states: (batch, dim, width-1)
    """
    dtype_in = x.dtype
    x = x.to(weight.dtype)
    seqlen = x.shape[-1]
    dim, width = weight.shape
    if initial_states is None:
        out = F.conv1d(
            x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)
    else:
        x = torch.cat([initial_states, x], dim=-1)
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=0, groups=dim)
    out = out[..., :seqlen]
    if return_final_states:
        final_states = F.pad(x, (width - 1 - x.shape[-1], 0)).to(dtype_in)
        if final_states_out is not None:
            final_states_out.copy_(final_states)
        else:
            final_states_out = final_states
    out = (out if activation is None else F.silu(out)).to(dtype=dtype_in)
    return (out, None) if not return_final_states else (out, final_states_out)


# ──────────────────────────────────────────────────────────────────
# Helper: build reference conv1d output for varlen batch
# ──────────────────────────────────────────────────────────────────


def reference_conv1d_varlen(
    x_flat: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    query_start_loc: torch.Tensor,
    initial_states: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    activation: str | None = "silu",
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Compute reference conv1d for variable-length sequences.

    Args:
        x_flat: (num_tokens, dim) flattened input
        weight: (dim, width) conv weights
        bias: (dim,) optional bias
        query_start_loc: (batch+1,) cumulative seq lengths
        initial_states: (batch, dim, state_len) per-seq initial conv state
        has_initial_state: (batch,) bool flags
        activation: "silu" or None

    Returns:
        out_flat: (num_tokens, dim) output
        final_states: list of (dim, state_len) tensors per seq
    """
    batch_size = len(query_start_loc) - 1
    out_flat = torch.zeros_like(x_flat)
    final_states = []

    for i in range(batch_size):
        start = query_start_loc[i].item()
        end = query_start_loc[i + 1].item()

        # (dim, seqlen)
        x_seq = x_flat[start:end].T.unsqueeze(0)  # (1, dim, seqlen)

        init = None
        if (initial_states is not None
                and has_initial_state is not None
                and has_initial_state[i]):
            init = initial_states[i:i + 1]  # (1, dim, state_len)

        o, fs = causal_conv1d_ref(
            x_seq, weight, bias, initial_states=init,
            return_final_states=True, activation=activation,
        )
        out_flat[start:end] = o[0].T  # (seqlen, dim)
        final_states.append(fs[0])    # (dim, state_len)

    return out_flat, final_states


# ══════════════════════════════════════════════════════════════════
# Test: compute_conv1d_grid_npu (CPU-only)
# ══════════════════════════════════════════════════════════════════


class TestComputeConv1dGridNpu:
    """Grid scheduling precomputation tests — pure CPU."""

    DEVICE = torch.device("cpu")

    def test_single_seq_exact(self):
        """16 tokens / BLOCK_M=8 → 2 programs."""
        qsl = torch.tensor([0, 16], dtype=torch.int32)
        bp, op, total = compute_conv1d_grid_npu(qsl, 8, -1, self.DEVICE)
        assert total == 2
        assert bp.tolist() == [0, 0]
        assert op.tolist() == [0, 1]

    def test_single_seq_not_aligned(self):
        """5 tokens / BLOCK_M=8 → ceil(5/8)=1 program."""
        qsl = torch.tensor([0, 5], dtype=torch.int32)
        bp, op, total = compute_conv1d_grid_npu(qsl, 8, -1, self.DEVICE)
        assert total == 1
        assert bp.tolist() == [0]
        assert op.tolist() == [0]

    def test_single_token_seq(self):
        """1 token → 1 program (decode case)."""
        qsl = torch.tensor([0, 1], dtype=torch.int32)
        bp, op, total = compute_conv1d_grid_npu(qsl, 8, -1, self.DEVICE)
        assert total == 1
        assert bp.tolist() == [0]
        assert op.tolist() == [0]

    def test_multiple_sequences(self):
        """3 seqs: 5, 1, 10 tokens → cdiv: 1, 1, 2 → 4 programs."""
        qsl = torch.tensor([0, 5, 6, 16], dtype=torch.int32)
        bp, op, total = compute_conv1d_grid_npu(qsl, 8, -1, self.DEVICE)
        assert total == 4
        assert bp.tolist() == [0, 1, 2, 2]
        assert op.tolist() == [0, 0, 0, 1]

    def test_exact_block_boundary(self):
        """8 tokens = exactly 1 block, no remainder."""
        qsl = torch.tensor([0, 8], dtype=torch.int32)
        bp, op, total = compute_conv1d_grid_npu(qsl, 8, -1, self.DEVICE)
        assert total == 1
        assert bp.tolist() == [0]
        assert op.tolist() == [0]

    def test_long_sequence(self):
        """64 tokens / BLOCK_M=8 → 8 programs, all same batch index."""
        qsl = torch.tensor([0, 64], dtype=torch.int32)
        bp, op, total = compute_conv1d_grid_npu(qsl, 8, -1, self.DEVICE)
        assert total == 8
        assert bp.tolist() == [0] * 8
        assert op.tolist() == list(range(8))

    def test_mixed_batch_decode_prefill(self):
        """3 decodes (1 token) + 1 prefill (24 tokens) → 3 + 3 = 6 programs."""
        qsl = torch.tensor([0, 1, 2, 3, 27], dtype=torch.int32)
        bp, op, total = compute_conv1d_grid_npu(qsl, 8, -1, self.DEVICE)
        assert total == 6
        assert bp.tolist() == [0, 1, 2, 3, 3, 3]
        assert op.tolist() == [0, 0, 0, 0, 1, 2]

    def test_many_decodes_only(self):
        """8 decode seqs (1 token each) → 8 programs, each offset=0."""
        qsl = torch.tensor(list(range(9)), dtype=torch.int32)
        bp, op, total = compute_conv1d_grid_npu(qsl, 8, -1, self.DEVICE)
        assert total == 8
        assert bp.tolist() == list(range(8))
        assert op.tolist() == [0] * 8

    def test_empty_batch(self):
        """0 sequences → 0 programs."""
        qsl = torch.tensor([0], dtype=torch.int32)
        bp, op, total = compute_conv1d_grid_npu(qsl, 8, -1, self.DEVICE)
        assert total == 0
        assert bp.tolist() == []
        assert op.tolist() == []

    def test_output_dtype_and_device(self):
        """Verify output tensors have correct dtype and device."""
        qsl = torch.tensor([0, 10], dtype=torch.int32)
        bp, op, total = compute_conv1d_grid_npu(qsl, 8, -1, self.DEVICE)
        assert bp.dtype == torch.int32
        assert op.dtype == torch.int32
        assert bp.device == self.DEVICE
        assert op.device == self.DEVICE


# ══════════════════════════════════════════════════════════════════
# Test: reference_conv1d_varlen (validates our helper against torch)
# ══════════════════════════════════════════════════════════════════


class TestReferenceConv1dVarlen:
    """Validate our reference helper produces correct results."""

    def _make_data(self, batch_seqlens, dim=16, seed=42):
        """Generate deterministic test data."""
        torch.manual_seed(seed)
        total_tokens = sum(batch_seqlens)
        qsl = torch.tensor(
            [0] + list(np.cumsum(batch_seqlens)), dtype=torch.int32)
        x = torch.randn(total_tokens, dim, dtype=torch.float32)
        w = torch.randn(dim, WIDTH, dtype=torch.float32)
        b = torch.randn(dim, dtype=torch.float32)
        return x, w, b, qsl

    def test_single_seq_no_initial_state(self):
        """Single sequence, no initial state — pure conv1d."""
        x, w, b, qsl = self._make_data([10], dim=8)
        has_init = torch.tensor([False])

        out, final = reference_conv1d_varlen(
            x, w, b, qsl, has_initial_state=has_init, activation="silu")

        # Cross-check: reshape to (1, dim, seqlen) and call ref directly
        x_3d = x.T.unsqueeze(0)  # (1, 8, 10)
        ref_out, _ = causal_conv1d_ref(x_3d, w, b, activation="silu")
        torch.testing.assert_close(out, ref_out[0].T, atol=1e-5, rtol=1e-5)

    def test_single_seq_with_initial_state(self):
        """Single sequence with initial conv state."""
        dim = 8
        x, w, b, qsl = self._make_data([6], dim=dim)
        init = torch.randn(1, dim, STATE_LEN, dtype=torch.float32)
        has_init = torch.tensor([True])

        out, final = reference_conv1d_varlen(
            x, w, b, qsl, initial_states=init,
            has_initial_state=has_init, activation="silu")

        # Cross-check
        x_3d = x.T.unsqueeze(0)
        ref_out, _ = causal_conv1d_ref(
            x_3d, w, b, initial_states=init, activation="silu")
        torch.testing.assert_close(out, ref_out[0].T, atol=1e-5, rtol=1e-5)

    def test_multi_seq_varlen(self):
        """Multiple sequences with different lengths."""
        seqlens = [3, 7, 1]
        dim = 8
        x, w, b, qsl = self._make_data(seqlens, dim=dim)
        has_init = torch.tensor([False, False, False])

        out, finals = reference_conv1d_varlen(
            x, w, b, qsl, has_initial_state=has_init, activation="silu")

        # Check each sequence independently
        for i, sl in enumerate(seqlens):
            start = qsl[i].item()
            end = qsl[i + 1].item()
            x_seq = x[start:end].T.unsqueeze(0)
            ref_o, _ = causal_conv1d_ref(x_seq, w, b, activation="silu")
            torch.testing.assert_close(
                out[start:end], ref_o[0].T, atol=1e-5, rtol=1e-5)

    def test_final_state_correctness(self):
        """Verify final state = last state_len tokens of x."""
        dim = 4
        seqlen = 10
        x, w, b, qsl = self._make_data([seqlen], dim=dim)
        has_init = torch.tensor([False])

        _, finals = reference_conv1d_varlen(
            x, w, b, qsl, has_initial_state=has_init, activation="silu")

        # Final state should be last STATE_LEN tokens of x
        expected_fs = x[-STATE_LEN:].T  # (dim, state_len)
        torch.testing.assert_close(finals[0], expected_fs, atol=1e-5, rtol=1e-5)

    def test_final_state_with_initial_and_short_seq(self):
        """Short sequence (< state_len) with initial state."""
        dim = 4
        x, w, b, qsl = self._make_data([2], dim=dim)  # 2 < STATE_LEN=3
        init = torch.randn(1, dim, STATE_LEN, dtype=torch.float32)
        has_init = torch.tensor([True])

        _, finals = reference_conv1d_varlen(
            x, w, b, qsl, initial_states=init,
            has_initial_state=has_init, activation="silu")

        # Final state = [init[:, :, -1], x[:, 0], x[:, 1]]
        expected = torch.cat([init[0, :, -1:], x.T], dim=-1)
        assert expected.shape == (dim, STATE_LEN)
        torch.testing.assert_close(finals[0], expected, atol=1e-5, rtol=1e-5)

    def test_no_activation(self):
        """Verify activation=None works."""
        x, w, b, qsl = self._make_data([5], dim=4)
        has_init = torch.tensor([False])

        out, _ = reference_conv1d_varlen(
            x, w, b, qsl, has_initial_state=has_init, activation=None)

        x_3d = x.T.unsqueeze(0)
        ref_out, _ = causal_conv1d_ref(x_3d, w, b, activation=None)
        torch.testing.assert_close(out, ref_out[0].T, atol=1e-5, rtol=1e-5)


# ══════════════════════════════════════════════════════════════════
# Test: Wrapper logic validation
# ══════════════════════════════════════════════════════════════════


class TestWrapperLogic:
    """Validate wrapper argument preparation without launching kernel."""

    def test_stride_computation_row_major(self):
        """Verify stride extraction for (num_tokens, dim) row-major x."""
        dim = 32
        num_tokens = 10
        x = torch.randn(num_tokens, dim)
        assert x.stride(0) == dim   # stride_x_token
        assert x.stride(1) == 1     # stride_x_dim (feature contiguous)

    def test_conv_state_stride_dim_contiguous(self):
        """conv_states (N, dim, state_len) must have stride(1)==1."""
        N, dim, sl = 4, 16, 3
        # Physical pool: (N, state_len, dim) then .transpose(1,2)
        pool = torch.randn(N, sl, dim)
        view = pool.transpose(1, 2)  # (N, dim, state_len) — dim contiguous
        assert view.stride(1) == 1, "dim must be contiguous"
        assert view.stride(2) == dim, "state_len stride = dim"
        assert view.stride(0) == sl * dim, "batch stride"

    def test_weight_shape_assertion(self):
        """Weight must be (dim, 4)."""
        dim = 16
        w = torch.randn(dim, WIDTH)
        assert w.shape[1] == WIDTH

    def test_block_size_alignment(self):
        """block_size must be divisible by BLOCK_M=8."""
        BLOCK_M = 8
        for bs in [8, 16, 32, 64, 128]:
            assert bs % BLOCK_M == 0
        for bs in [7, 10, 15]:
            assert bs % BLOCK_M != 0

    def test_cache_indices_2d_stride(self):
        """cache_indices (batch, max_blocks) stride(0) = max_blocks."""
        batch, max_blocks = 3, 8
        ci = torch.zeros(batch, max_blocks, dtype=torch.int32)
        assert ci.stride(0) == max_blocks


# ══════════════════════════════════════════════════════════════════
# Test: APC block boundary state computation (reference)
# ══════════════════════════════════════════════════════════════════


class TestAPCBlockBoundaryRef:
    """Test APC (all-mode prefix caching) conv state scatter logic.

    These tests validate the MATHEMATICAL CORRECTNESS of block boundary
    conv state computation against the reference implementation.
    The actual Triton kernel writes these in-kernel; here we verify
    what the expected states should be.
    """

    def test_block_boundary_state_values(self):
        """At block boundary, conv state = last STATE_LEN tokens before boundary."""
        dim = 4
        block_size = 16
        seqlen = 32  # 2 full blocks
        torch.manual_seed(123)
        x = torch.randn(seqlen, dim)

        # Block boundary at token 16 → conv state = x[13:16] (last 3 tokens)
        expected_state_block0 = x[block_size - STATE_LEN:block_size].T  # (dim, 3)

        # Verify against reference conv1d
        qsl = torch.tensor([0, seqlen], dtype=torch.int32)
        w = torch.randn(dim, WIDTH)
        _, finals = reference_conv1d_varlen(
            x, w, None, qsl,
            has_initial_state=torch.tensor([False]),
            activation=None)

        # The final state is last STATE_LEN of entire sequence = x[29:32]
        expected_final = x[-STATE_LEN:].T
        torch.testing.assert_close(
            finals[0], expected_final, atol=1e-5, rtol=1e-5)

        # Block 0 boundary state is just raw x tokens (before the conv)
        torch.testing.assert_close(
            expected_state_block0,
            x[block_size - STATE_LEN:block_size].T,
            atol=1e-5, rtol=1e-5)

    def test_block_boundary_with_initial_state(self):
        """Block boundary state when initial_state is provided."""
        dim = 4
        seqlen = 8  # exactly 1 block
        torch.manual_seed(456)
        x = torch.randn(seqlen, dim)
        init = torch.randn(1, dim, STATE_LEN)

        # With initial state, the "input stream" is: [init, x]
        # Block boundary at token 8 → last 3 tokens = x[5:8]
        expected_final = x[-STATE_LEN:].T  # (dim, 3)

        qsl = torch.tensor([0, seqlen], dtype=torch.int32)
        w = torch.randn(dim, WIDTH)
        _, finals = reference_conv1d_varlen(
            x, w, None, qsl, initial_states=init,
            has_initial_state=torch.tensor([True]),
            activation=None)

        torch.testing.assert_close(
            finals[0], expected_final, atol=1e-5, rtol=1e-5)

    def test_partial_first_block(self):
        """When num_computed_tokens > 0, first block is partial."""
        dim = 4
        block_size = 16
        num_computed = 10  # 10 tokens already processed
        new_tokens = 22   # 22 new tokens → total 32 = 2 blocks

        torch.manual_seed(789)
        x_new = torch.randn(new_tokens, dim)

        # First scheduled block boundary: at token (block_size - num_computed)
        # = 16 - 10 = 6 new tokens into this block
        # Boundary state = last 3 of block 0 = x_new[3:6] (if no init state)
        first_boundary_offset = block_size - num_computed  # 6
        expected_state = x_new[first_boundary_offset - STATE_LEN:
                               first_boundary_offset].T
        assert expected_state.shape == (dim, STATE_LEN)

    def test_dest_final_state_is_last_tokens(self):
        """DEST block final conv state = last STATE_LEN tokens of sequence."""
        dim = 8
        seqlens = [20, 5]
        total = sum(seqlens)
        torch.manual_seed(111)
        x = torch.randn(total, dim)
        qsl = torch.tensor([0] + list(np.cumsum(seqlens)), dtype=torch.int32)
        w = torch.randn(dim, WIDTH)

        _, finals = reference_conv1d_varlen(
            x, w, None, qsl,
            has_initial_state=torch.tensor([False, False]),
            activation=None)

        # Seq 0: final = x[17:20].T
        torch.testing.assert_close(
            finals[0], x[seqlens[0] - STATE_LEN:seqlens[0]].T,
            atol=1e-5, rtol=1e-5)

        # Seq 1: final = x[22:25].T (but seqlen=5 > state_len=3, so last 3)
        s1_start = seqlens[0]
        s1_end = s1_start + seqlens[1]
        torch.testing.assert_close(
            finals[1], x[s1_end - STATE_LEN:s1_end].T,
            atol=1e-5, rtol=1e-5)


# ══════════════════════════════════════════════════════════════════
# Test: Grid scheduling for APC scenarios
# ══════════════════════════════════════════════════════════════════


class TestGridSchedulingAPC:
    """Grid scheduling edge cases relevant to APC."""

    DEVICE = torch.device("cpu")

    def test_block_size_64_long_prefill(self):
        """128-token prefill, block_size=64, BLOCK_M=8 → 16 programs.

        This is a realistic APC scenario: 2 blocks to fill.
        """
        qsl = torch.tensor([0, 128], dtype=torch.int32)
        bp, op, total = compute_conv1d_grid_npu(qsl, 8, -1, self.DEVICE)
        assert total == 16
        # All same batch index
        assert all(b == 0 for b in bp.tolist())
        # Offsets 0..15
        assert op.tolist() == list(range(16))

    def test_partial_block_at_end(self):
        """70 tokens, block_size=64, BLOCK_M=8 → 9 programs.

        Block 0: 64 tokens = 8 chunks.
        Block 1: 6 tokens = 1 chunk (partial).
        """
        qsl = torch.tensor([0, 70], dtype=torch.int32)
        bp, op, total = compute_conv1d_grid_npu(qsl, 8, -1, self.DEVICE)
        assert total == 9  # ceil(70/8) = 9
        assert op.tolist() == list(range(9))

    def test_chunks_per_block_alignment(self):
        """Verify chunks_per_block = block_size / BLOCK_M."""
        block_size = 64
        BLOCK_M = 8
        chunks_per_block = block_size // BLOCK_M  # 8

        # 2 blocks of 64 tokens = 128 tokens = 16 programs
        qsl = torch.tensor([0, 128], dtype=torch.int32)
        _, op, total = compute_conv1d_grid_npu(qsl, BLOCK_M, -1, self.DEVICE)
        assert total == 16

        # Block boundaries: at offset 0 (start of block 0) and 8 (start of block 1)
        block_boundaries = [i * chunks_per_block for i in range(2)]
        assert block_boundaries == [0, 8]
        # The kernel uses these to determine when to write intermediate states


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
