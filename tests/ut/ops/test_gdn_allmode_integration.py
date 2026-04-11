# SPDX-License-Identifier: Apache-2.0
"""P6 Integration tests: All-mode forward state management flow.

Tests the complete state lifecycle:
  _build_initial_state → kernel (mocked) → _write_final_states → _scatter_intermediate_states

Validates that pool states are correctly read/transformed/written across
mixed-batch scenarios. All CPU, no NPU dependency.
"""

from types import SimpleNamespace

import pytest
import torch

# ──────────────────────────────────────────────────────────────────
# Inline helpers (same as in gdn.py, avoids torch_npu import chain)
# ──────────────────────────────────────────────────────────────────


def _build_initial_state(ssm_state, metadata, num_decodes, num_prefills,
                         transpose_state=True):
    num_seqs = num_decodes + num_prefills
    initial = ssm_state.new_zeros(num_seqs, *ssm_state.shape[1:])
    source_slots = metadata.block_state_indices
    if num_decodes > 0:
        d_slots = source_slots[:num_decodes]
        valid = d_slots >= 0
        if valid.any():
            state = ssm_state[d_slots[valid].long()]
            if transpose_state:
                state = state.transpose(-1, -2).contiguous()
            initial[:num_decodes][valid] = state
    if num_prefills > 0:
        p_slots = source_slots[num_decodes:]
        valid = p_slots >= 0
        if valid.any():
            state = ssm_state[p_slots[valid].long()]
            if transpose_state:
                state = state.transpose(-1, -2).contiguous()
            initial[num_decodes:][valid] = state
        has_init = metadata.has_initial_state
        if has_init is not None:
            no_init = ~has_init[num_decodes:]
            initial[num_decodes:][no_init] = 0
    return initial


def _write_final_states(ssm_state, final_state, metadata, num_decodes,
                        transpose_state=True):
    dest_slots = metadata.non_spec_state_indices_tensor
    if num_decodes > 0:
        d_dest = dest_slots[:num_decodes]
        valid = d_dest >= 0
        if valid.any():
            state = final_state[:num_decodes][valid].to(ssm_state.dtype)
            if transpose_state:
                state = state.transpose(-1, -2).contiguous()
            ssm_state[d_dest[valid].long()] = state
    num_prefills = final_state.shape[0] - num_decodes
    if num_prefills > 0:
        p_dest = dest_slots[num_decodes:]
        valid = p_dest >= 0
        if valid.any():
            state = final_state[num_decodes:][valid].to(ssm_state.dtype)
            if transpose_state:
                state = state.transpose(-1, -2).contiguous()
            ssm_state[p_dest[valid].long()] = state


def _scatter_intermediate_states(ssm_state, chunk_history, metadata,
                                 num_decodes, transpose_state=True):
    block_size = metadata.mamba_block_size
    chunk_size = metadata.all_mode_chunk_size
    chunks_per_block = block_size // chunk_size
    prefill_chunk_start = metadata.prefill_chunk_start
    prefill_offsets = metadata.prefill_chunk_offsets
    if prefill_offsets is None:
        return
    num_prefills = len(prefill_offsets) - 1
    block_table = metadata.block_table_2d[num_decodes:]
    first_sched = metadata.block_idx_first_scheduled_token[num_decodes:]
    last_sched = metadata.block_idx_last_scheduled_token[num_decodes:]
    num_comp = metadata.num_computed_tokens_all[num_decodes:]
    for seq_idx in range(num_prefills):
        chunk_start = prefill_offsets[seq_idx].item()
        block_first = first_sched[seq_idx].item()
        block_last = last_sched[seq_idx].item()
        n_blocks = block_last - block_first
        if n_blocks <= 0:
            continue
        cache_slots = block_table[seq_idx, block_first:block_last]
        valid = cache_slots >= 0
        first_chunk = prefill_chunk_start + chunk_start
        first_aligned_chunk = first_chunk + chunks_per_block - 1
        num_unaligned = num_comp[seq_idx].item() % block_size
        if num_unaligned > 0:
            first_aligned_chunk -= num_unaligned // chunk_size
        states = chunk_history[
            first_aligned_chunk:
            first_aligned_chunk + n_blocks * chunks_per_block:
            chunks_per_block
        ]
        write_states = states[:valid.sum()].to(ssm_state.dtype)
        if transpose_state:
            write_states = write_states.transpose(-1, -2).contiguous()
        ssm_state[cache_slots[valid].long()] = write_states


# ──────────────────────────────────────────────────────────────────
# Test fixtures
# ──────────────────────────────────────────────────────────────────

H, K, V = 2, 4, 4
POOL_SIZE = 32
BLOCK_SIZE = 64
CHUNK_SIZE = 64


def _make_metadata(**kw):
    m = SimpleNamespace()
    m.mamba_block_size = kw.get('block_size', BLOCK_SIZE)
    m.all_mode_chunk_size = kw.get('chunk_size', CHUNK_SIZE)
    m.block_state_indices = torch.tensor(kw['source_slots'], dtype=torch.int32)
    m.non_spec_state_indices_tensor = torch.tensor(
        kw['dest_slots'], dtype=torch.int32)
    n = len(kw['source_slots'])
    m.has_initial_state = torch.tensor(
        kw.get('has_initial', [True] * n), dtype=torch.bool)
    m.block_table_2d = kw.get('block_table',
                               torch.full((n, 4), -1, dtype=torch.int32))
    m.block_idx_first_scheduled_token = torch.tensor(
        kw.get('first_sched', [0] * n), dtype=torch.int32)
    m.block_idx_last_scheduled_token = torch.tensor(
        kw.get('last_sched', [0] * n), dtype=torch.int32)
    m.num_computed_tokens_all = torch.tensor(
        kw.get('num_computed', [0] * n), dtype=torch.int32)
    m.prefill_chunk_start = kw.get('prefill_chunk_start', 0)
    offsets = kw.get('prefill_chunk_offsets', None)
    m.prefill_chunk_offsets = (torch.tensor(offsets, dtype=torch.long)
                               if offsets is not None else None)
    return m


def mock_chunk_kernel(initial_state, num_chunks):
    """Simulate chunk_gated_delta_rule: returns (output, final_state, chunk_history).

    - output: dummy tensor
    - final_state: initial_state + 1.0 (deterministic transform)
    - chunk_history: [total_chunks, H, K, V] with sequential values
    """
    B = initial_state.shape[0]
    output = torch.zeros(1)  # placeholder
    final_state = initial_state + 1.0
    chunk_history = torch.arange(
        num_chunks * H * K * V, dtype=initial_state.dtype
    ).reshape(num_chunks, H, K, V)
    return output, final_state, chunk_history


# ══════════════════════════════════════════════════════════════════
# Integration test 1: Full SSM flow (Qwen3Next — with transpose)
# ══════════════════════════════════════════════════════════════════


class TestSSMFlowQwen3Next:
    """End-to-end: build → kernel → write → scatter for Qwen3Next (transpose)."""

    def test_prefill_only_single_block(self):
        """1 prefill seq, 64 tokens (1 block). No intermediate scatter needed."""
        torch.manual_seed(100)
        pool = torch.randn(POOL_SIZE, H, V, K)  # Qwen3Next: pool [V,K]
        pool_orig = pool.clone()

        source_slot, dest_slot = 3, 7
        m = _make_metadata(
            source_slots=[source_slot], dest_slots=[dest_slot],
            has_initial=[True],
            first_sched=[0], last_sched=[0],  # 0 blocks to scatter
            num_computed=[0],
            prefill_chunk_offsets=[0, 1],
            prefill_chunk_start=0,
        )

        # Step 1: Build initial state
        initial = _build_initial_state(pool, m, 0, 1, transpose_state=True)
        # Should be pool[3] transposed [V,K] -> [K,V]
        expected_init = pool_orig[source_slot].transpose(-1, -2).contiguous()
        torch.testing.assert_close(initial[0], expected_init)

        # Step 2: Mock kernel
        _, final_state, chunk_history = mock_chunk_kernel(initial, 1)

        # Step 3: Write final states
        _write_final_states(pool, final_state, m, 0, transpose_state=True)
        # final_state[0] transposed [K,V] -> [V,K] should be at pool[dest_slot]
        expected_written = final_state[0].transpose(-1, -2).contiguous()
        torch.testing.assert_close(pool[dest_slot], expected_written)

        # Step 4: Scatter (0 blocks → no-op)
        pool_before_scatter = pool.clone()
        _scatter_intermediate_states(pool, chunk_history, m, 0,
                                     transpose_state=True)
        torch.testing.assert_close(pool, pool_before_scatter)

    def test_prefill_only_three_blocks(self):
        """1 prefill, 192 tokens → 3 blocks. Scatter 2 intermediate boundaries."""
        torch.manual_seed(101)
        pool = torch.randn(POOL_SIZE, H, V, K)

        block_table = torch.tensor([[10, 11, 12, -1]], dtype=torch.int32)
        m = _make_metadata(
            source_slots=[-1], dest_slots=[12],  # no prior state, dest=last block
            has_initial=[False],  # fresh prefill
            block_table=block_table,
            first_sched=[0], last_sched=[2],  # scatter blocks 0, 1
            num_computed=[0],
            prefill_chunk_offsets=[0, 3],
            prefill_chunk_start=0,
        )

        # Step 1: Build initial
        initial = _build_initial_state(pool, m, 0, 1, transpose_state=True)
        # has_initial=False → zeroed
        assert (initial[0] == 0).all()

        # Step 2: Kernel
        _, final_state, chunk_history = mock_chunk_kernel(initial, 3)

        # Step 3: Write final
        _write_final_states(pool, final_state, m, 0, transpose_state=True)
        expected_final = final_state[0].transpose(-1, -2).contiguous()
        torch.testing.assert_close(pool[12], expected_final)

        # Step 4: Scatter intermediate
        _scatter_intermediate_states(pool, chunk_history, m, 0,
                                     transpose_state=True)
        # Block 0 (slot 10) gets chunk_history[0] transposed
        expected_b0 = chunk_history[0].transpose(-1, -2).contiguous()
        torch.testing.assert_close(pool[10], expected_b0)
        # Block 1 (slot 11) gets chunk_history[1] transposed
        expected_b1 = chunk_history[1].transpose(-1, -2).contiguous()
        torch.testing.assert_close(pool[11], expected_b1)

    def test_mixed_batch_decode_plus_prefill(self):
        """2 decodes + 1 prefill (128 tokens, 2 blocks). Full mixed batch."""
        torch.manual_seed(102)
        pool = torch.randn(POOL_SIZE, H, V, K)
        pool_orig = pool.clone()

        # Decode 0: source=5, dest=6
        # Decode 1: source=8, dest=9
        # Prefill 0: source=1 (has context), dest=15, blocks [20, 21]
        block_table = torch.tensor(
            [[6, -1, -1, -1],    # decode 0
             [9, -1, -1, -1],    # decode 1
             [20, 21, -1, -1]],  # prefill 0
            dtype=torch.int32)
        m = _make_metadata(
            source_slots=[5, 8, 1], dest_slots=[6, 9, 21],
            has_initial=[True, True, True],
            block_table=block_table,
            first_sched=[0, 0, 0], last_sched=[0, 0, 1],
            num_computed=[60, 63, 0],
            prefill_chunk_offsets=[0, 2],
            prefill_chunk_start=2,  # 2 decode chunks before prefill
        )

        # Step 1: Build initial (2 decodes + 1 prefill)
        initial = _build_initial_state(pool, m, 2, 1, transpose_state=True)
        assert initial.shape == (3, H, V, K)
        # Decode 0: pool[5] transposed
        torch.testing.assert_close(
            initial[0], pool_orig[5].transpose(-1, -2).contiguous())
        # Decode 1: pool[8] transposed
        torch.testing.assert_close(
            initial[1], pool_orig[8].transpose(-1, -2).contiguous())
        # Prefill 0: pool[1] transposed
        torch.testing.assert_close(
            initial[2], pool_orig[1].transpose(-1, -2).contiguous())

        # Step 2: Kernel (2 decode + 2 prefill chunks = 4 total)
        _, final_state, chunk_history = mock_chunk_kernel(initial, 4)

        # Step 3: Write final
        _write_final_states(pool, final_state, m, 2, transpose_state=True)
        # Decode 0 → slot 6
        torch.testing.assert_close(
            pool[6], final_state[0].transpose(-1, -2).contiguous())
        # Decode 1 → slot 9
        torch.testing.assert_close(
            pool[9], final_state[1].transpose(-1, -2).contiguous())
        # Prefill 0 → slot 21
        torch.testing.assert_close(
            pool[21], final_state[2].transpose(-1, -2).contiguous())

        # Step 4: Scatter (prefill: block 0, scatter 1 boundary)
        _scatter_intermediate_states(pool, chunk_history, m, 2,
                                     transpose_state=True)
        # prefill_chunk_start=2, chunk_start=0, first_chunk=2
        # first_aligned_chunk = 2+1-1 = 2, num_computed=0, no shift
        # states = chunk_history[2:2+1:1] = chunk_history[2]
        # Written to block_table[2(prefill row 0), 0] = slot 20
        expected_b0 = chunk_history[2].transpose(-1, -2).contiguous()
        torch.testing.assert_close(pool[20], expected_b0)


# ══════════════════════════════════════════════════════════════════
# Integration test 2: Full SSM flow (Qwen3.5 — no transpose)
# ══════════════════════════════════════════════════════════════════


class TestSSMFlowQwen35:
    """End-to-end: build → kernel → write → scatter for Qwen3.5 (no transpose)."""

    def test_prefill_two_blocks_no_transpose(self):
        """Qwen3.5: pool [K,V] == kernel [K,V], no transpose needed."""
        torch.manual_seed(200)
        pool = torch.randn(POOL_SIZE, H, K, V)
        pool_orig = pool.clone()

        block_table = torch.tensor([[5, 6, -1, -1]], dtype=torch.int32)
        m = _make_metadata(
            source_slots=[5], dest_slots=[6],
            has_initial=[True],
            block_table=block_table,
            first_sched=[0], last_sched=[1],
            num_computed=[0],
            prefill_chunk_offsets=[0, 2],
            prefill_chunk_start=0,
        )

        # Build — no transpose
        initial = _build_initial_state(pool, m, 0, 1, transpose_state=False)
        torch.testing.assert_close(initial[0], pool_orig[5])

        # Kernel
        _, final, history = mock_chunk_kernel(initial, 2)

        # Write — no transpose
        _write_final_states(pool, final, m, 0, transpose_state=False)
        torch.testing.assert_close(pool[6], final[0])

        # Scatter — no transpose
        _scatter_intermediate_states(pool, history, m, 0,
                                     transpose_state=False)
        torch.testing.assert_close(pool[5], history[0])

    def test_decode_only_no_scatter(self):
        """Decode-only: read SOURCE, transform, write DEST. No scatter."""
        torch.manual_seed(201)
        pool = torch.randn(POOL_SIZE, H, K, V)
        pool_orig = pool.clone()

        m = _make_metadata(
            source_slots=[2, 4], dest_slots=[3, 5],
            has_initial=[True, True])

        # Build (decode-only, no prefills)
        initial = _build_initial_state(pool, m, 2, 0, transpose_state=False)
        torch.testing.assert_close(initial[0], pool_orig[2])
        torch.testing.assert_close(initial[1], pool_orig[4])

        # Kernel (1 chunk per decode)
        _, final, history = mock_chunk_kernel(initial, 2)

        # Write
        _write_final_states(pool, final, m, 2, transpose_state=False)
        torch.testing.assert_close(pool[3], final[0])
        torch.testing.assert_close(pool[5], final[1])


# ══════════════════════════════════════════════════════════════════
# Integration test 3: SOURCE ≠ DEST validation
# ══════════════════════════════════════════════════════════════════


class TestSourceDestSeparation:
    """Verify that reads use SOURCE slots and writes use DEST slots."""

    def test_source_and_dest_are_different_slots(self):
        """Read from source=2, write to dest=10. Source unchanged after write."""
        torch.manual_seed(300)
        pool = torch.randn(POOL_SIZE, H, K, V)
        pool_orig = pool.clone()

        m = _make_metadata(
            source_slots=[2], dest_slots=[10])

        initial = _build_initial_state(pool, m, 1, 0, transpose_state=False)
        # Read from 2
        torch.testing.assert_close(initial[0], pool_orig[2])

        _, final, _ = mock_chunk_kernel(initial, 1)
        _write_final_states(pool, final, m, 1, transpose_state=False)

        # Source slot 2 should be UNCHANGED
        torch.testing.assert_close(pool[2], pool_orig[2])
        # Dest slot 10 should have new data
        torch.testing.assert_close(pool[10], final[0])

    def test_same_source_dest_works(self):
        """When source==dest (like align-mode), still works."""
        torch.manual_seed(301)
        pool = torch.randn(POOL_SIZE, H, K, V)

        m = _make_metadata(
            source_slots=[5], dest_slots=[5])

        initial = _build_initial_state(pool, m, 1, 0, transpose_state=False)
        _, final, _ = mock_chunk_kernel(initial, 1)
        _write_final_states(pool, final, m, 1, transpose_state=False)

        # pool[5] should have the new data
        torch.testing.assert_close(pool[5], final[0])


# ══════════════════════════════════════════════════════════════════
# Integration test 4: Edge cases
# ══════════════════════════════════════════════════════════════════


class TestEdgeCases:

    def test_all_invalid_slots(self):
        """All slots are -1 → pool completely unchanged."""
        torch.manual_seed(400)
        pool = torch.randn(POOL_SIZE, H, K, V)
        pool_orig = pool.clone()

        m = _make_metadata(
            source_slots=[-1, -1], dest_slots=[-1, -1])

        initial = _build_initial_state(pool, m, 1, 1, transpose_state=False)
        assert (initial == 0).all()

        _, final, history = mock_chunk_kernel(initial, 2)
        _write_final_states(pool, final, m, 1, transpose_state=False)
        # No writes happened → pool unchanged
        torch.testing.assert_close(pool, pool_orig)

    def test_partial_invalid_slots(self):
        """Mix of valid and invalid slots."""
        torch.manual_seed(401)
        pool = torch.randn(POOL_SIZE, H, K, V)
        pool_orig = pool.clone()

        m = _make_metadata(
            source_slots=[-1, 3], dest_slots=[7, -1])

        # Build: slot 0 = -1 → zero, slot 1 = 3 → pool[3]
        initial = _build_initial_state(pool, m, 1, 1, transpose_state=False)
        assert (initial[0] == 0).all()
        torch.testing.assert_close(initial[1], pool_orig[3])

        _, final, _ = mock_chunk_kernel(initial, 2)

        # Write: dest 0 = 7 (valid), dest 1 = -1 (skip)
        _write_final_states(pool, final, m, 1, transpose_state=False)
        torch.testing.assert_close(pool[7], final[0])
        # All other slots unchanged
        for i in range(POOL_SIZE):
            if i != 7:
                torch.testing.assert_close(pool[i], pool_orig[i])

    def test_large_batch_many_prefills(self):
        """4 decodes + 3 prefills, multiple blocks per prefill."""
        torch.manual_seed(402)
        pool = torch.randn(POOL_SIZE, H, K, V)

        # 4 decodes: source=[0,1,2,3], dest=[4,5,6,7]
        # 3 prefills: source=[8,9,10], dest=[11,12,13]
        # Prefill 0: 128 tokens, 2 blocks [14,15], scatter 1
        # Prefill 1: 64 tokens, 1 block [16], scatter 0
        # Prefill 2: 192 tokens, 3 blocks [17,18,19], scatter 2
        block_table = torch.full((7, 4), -1, dtype=torch.int32)
        block_table[4, 0] = 14; block_table[4, 1] = 15
        block_table[5, 0] = 16
        block_table[6, 0] = 17; block_table[6, 1] = 18; block_table[6, 2] = 19

        m = _make_metadata(
            source_slots=[0, 1, 2, 3, 8, 9, 10],
            dest_slots=[4, 5, 6, 7, 15, 16, 19],
            has_initial=[True] * 7,
            block_table=block_table,
            first_sched=[0, 0, 0, 0, 0, 0, 0],
            last_sched=[0, 0, 0, 0, 1, 0, 2],
            num_computed=[50, 60, 30, 40, 0, 0, 0],
            # 4 decode chunks + (2+1+3) prefill chunks = 10 total
            prefill_chunk_offsets=[0, 2, 3, 6],
            prefill_chunk_start=4,
        )

        initial = _build_initial_state(pool, m, 4, 3, transpose_state=False)
        assert initial.shape[0] == 7

        _, final, history = mock_chunk_kernel(initial, 10)
        _write_final_states(pool, final, m, 4, transpose_state=False)
        _scatter_intermediate_states(pool, history, m, 4,
                                     transpose_state=False)

        # Verify decode final states written
        for i in range(4):
            torch.testing.assert_close(pool[4 + i], final[i])

        # Verify prefill final states
        torch.testing.assert_close(pool[15], final[4])
        torch.testing.assert_close(pool[16], final[5])
        torch.testing.assert_close(pool[19], final[6])

    def test_dtype_preservation(self):
        """Pool in bfloat16 → final state cast to bfloat16 on write."""
        pool = torch.randn(POOL_SIZE, H, K, V, dtype=torch.bfloat16)
        m = _make_metadata(source_slots=[0], dest_slots=[5])

        initial = _build_initial_state(pool, m, 1, 0, transpose_state=False)
        assert initial.dtype == torch.bfloat16

        # Kernel produces float32
        final = initial.float() + 1.0
        _write_final_states(pool, final, m, 1, transpose_state=False)
        # Should be cast back to bfloat16
        assert pool[5].dtype == torch.bfloat16

    def test_transpose_roundtrip_consistency(self):
        """Read with transpose + write with transpose = identity on layout."""
        torch.manual_seed(403)
        pool = torch.randn(POOL_SIZE, H, V, K)  # [V,K] layout
        original_state = pool[3].clone()

        m = _make_metadata(source_slots=[3], dest_slots=[3])

        # Read: pool[3] [V,K] → transpose → [K,V]
        initial = _build_initial_state(pool, m, 1, 0, transpose_state=True)
        # initial is [K,V]

        # "Kernel" does nothing (identity)
        final = initial.clone()

        # Write: [K,V] → transpose → [V,K] → pool[3]
        _write_final_states(pool, final, m, 1, transpose_state=True)

        # Pool[3] should match original (roundtrip)
        torch.testing.assert_close(pool[3], original_state)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
