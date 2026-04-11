# SPDX-License-Identifier: Apache-2.0
"""Unit tests for P4: GDN all-mode helper functions.

Tests _build_initial_state, _write_final_states, _scatter_intermediate_states.
These are pure-Python tests on CPU — no NPU/Triton dependency.
"""

from types import SimpleNamespace

import pytest
import torch


# ──────────────────────────────────────────────────────────────────
# Inlined helpers (from gdn.py — avoids importing torch_npu)
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
        first_aligned_chunk = first_chunk + chunks_per_block
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
# Fixtures and helpers
# ──────────────────────────────────────────────────────────────────


H, K, V = 2, 4, 4  # small dims for testing
POOL_SIZE = 16
BLOCK_SIZE = 64
CHUNK_SIZE = 64


def _make_pool(dtype=torch.float32, layout="VK"):
    """Create ssm_state pool [N, H, V, K] or [N, H, K, V]."""
    torch.manual_seed(42)
    if layout == "VK":
        return torch.randn(POOL_SIZE, H, V, K, dtype=dtype)
    else:
        return torch.randn(POOL_SIZE, H, K, V, dtype=dtype)


def _make_metadata(
    num_decodes: int,
    num_prefills: int,
    source_slots: list[int],
    dest_slots: list[int],
    has_initial: list[bool] | None = None,
    block_table: torch.Tensor | None = None,
    first_sched: list[int] | None = None,
    last_sched: list[int] | None = None,
    num_computed: list[int] | None = None,
    prefill_chunk_offsets: list[int] | None = None,
    prefill_chunk_start: int = 0,
):
    """Build SimpleNamespace metadata."""
    num_seqs = num_decodes + num_prefills
    m = SimpleNamespace()
    m.block_state_indices = torch.tensor(source_slots, dtype=torch.int32)
    m.non_spec_state_indices_tensor = torch.tensor(dest_slots, dtype=torch.int32)
    if has_initial is not None:
        m.has_initial_state = torch.tensor(has_initial, dtype=torch.bool)
    else:
        m.has_initial_state = torch.ones(num_seqs, dtype=torch.bool)
    m.mamba_block_size = BLOCK_SIZE
    m.all_mode_chunk_size = CHUNK_SIZE
    m.prefill_chunk_start = prefill_chunk_start
    if prefill_chunk_offsets is not None:
        m.prefill_chunk_offsets = torch.tensor(
            prefill_chunk_offsets, dtype=torch.long)
    else:
        m.prefill_chunk_offsets = None
    if block_table is not None:
        m.block_table_2d = block_table
    else:
        m.block_table_2d = torch.full(
            (num_seqs, 4), -1, dtype=torch.int32)
    if first_sched is not None:
        m.block_idx_first_scheduled_token = torch.tensor(
            first_sched, dtype=torch.int32)
    else:
        m.block_idx_first_scheduled_token = torch.zeros(
            num_seqs, dtype=torch.int32)
    if last_sched is not None:
        m.block_idx_last_scheduled_token = torch.tensor(
            last_sched, dtype=torch.int32)
    else:
        m.block_idx_last_scheduled_token = torch.zeros(
            num_seqs, dtype=torch.int32)
    if num_computed is not None:
        m.num_computed_tokens_all = torch.tensor(
            num_computed, dtype=torch.int32)
    else:
        m.num_computed_tokens_all = torch.zeros(
            num_seqs, dtype=torch.int32)
    return m


# ══════════════════════════════════════════════════════════════════
# Tests: _build_initial_state
# ══════════════════════════════════════════════════════════════════


class TestBuildInitialState:

    def test_decode_reads_source_with_transpose(self):
        """Decode seqs read from SOURCE slots, transpose [V,K]->[K,V]."""
        pool = _make_pool(layout="VK")  # [N, H, V, K]
        m = _make_metadata(
            num_decodes=2, num_prefills=0,
            source_slots=[3, 7], dest_slots=[5, 9])
        result = _build_initial_state(pool, m, 2, 0, transpose_state=True)
        assert result.shape == (2, H, V, K)
        # Result should be pool[3].transpose(-1,-2) for seq 0
        expected_0 = pool[3].transpose(-1, -2).contiguous()
        torch.testing.assert_close(result[0], expected_0)
        expected_1 = pool[7].transpose(-1, -2).contiguous()
        torch.testing.assert_close(result[1], expected_1)

    def test_prefill_reads_source_no_transpose(self):
        """Prefill without transpose (Qwen3.5 layout)."""
        pool = _make_pool(layout="KV")  # [N, H, K, V]
        m = _make_metadata(
            num_decodes=0, num_prefills=2,
            source_slots=[1, 4], dest_slots=[2, 5])
        result = _build_initial_state(pool, m, 0, 2, transpose_state=False)
        torch.testing.assert_close(result[0], pool[1])
        torch.testing.assert_close(result[1], pool[4])

    def test_invalid_source_slot_returns_zero(self):
        """Source slot = -1 → initial state is zero."""
        pool = _make_pool(layout="VK")
        m = _make_metadata(
            num_decodes=1, num_prefills=1,
            source_slots=[-1, 2], dest_slots=[3, 4])
        result = _build_initial_state(pool, m, 1, 1, transpose_state=True)
        # Decode seq 0: source=-1 → zero
        assert (result[0] == 0).all()
        # Prefill seq 0: source=2 → pool[2] transposed
        expected = pool[2].transpose(-1, -2).contiguous()
        torch.testing.assert_close(result[1], expected)

    def test_has_initial_state_false_zeros_prefill(self):
        """Prefill seq with has_initial_state=False → zeroed."""
        pool = _make_pool(layout="KV")
        m = _make_metadata(
            num_decodes=1, num_prefills=2,
            source_slots=[0, 1, 2], dest_slots=[3, 4, 5],
            has_initial=[True, True, False])  # decode, prefill0, prefill1
        result = _build_initial_state(pool, m, 1, 2, transpose_state=False)
        # Decode: has_initial doesn't matter (we always read)
        torch.testing.assert_close(result[0], pool[0])
        # Prefill 0: has_initial=True → pool[1]
        torch.testing.assert_close(result[1], pool[1])
        # Prefill 1: has_initial=False → zero
        assert (result[2] == 0).all()

    def test_mixed_batch(self):
        """Mixed batch: 2 decodes + 1 prefill."""
        pool = _make_pool(layout="VK")
        m = _make_metadata(
            num_decodes=2, num_prefills=1,
            source_slots=[5, 8, 3], dest_slots=[6, 9, 4])
        result = _build_initial_state(pool, m, 2, 1, transpose_state=True)
        assert result.shape == (3, H, V, K)


# ══════════════════════════════════════════════════════════════════
# Tests: _write_final_states
# ══════════════════════════════════════════════════════════════════


class TestWriteFinalStates:

    def test_decode_writes_to_dest_with_transpose(self):
        """Decode final state written to DEST with [K,V]->[V,K] transpose."""
        pool = torch.zeros(POOL_SIZE, H, V, K)
        final = torch.randn(2, H, K, V)  # kernel output [K,V]
        m = _make_metadata(
            num_decodes=2, num_prefills=0,
            source_slots=[1, 2], dest_slots=[5, 9])
        _write_final_states(pool, final, m, 2, transpose_state=True)
        expected_0 = final[0].transpose(-1, -2).contiguous()
        torch.testing.assert_close(pool[5], expected_0)
        expected_1 = final[1].transpose(-1, -2).contiguous()
        torch.testing.assert_close(pool[9], expected_1)

    def test_prefill_writes_to_dest_no_transpose(self):
        """Prefill writes without transpose (Qwen3.5)."""
        pool = torch.zeros(POOL_SIZE, H, K, V)
        final = torch.randn(1, H, K, V)
        m = _make_metadata(
            num_decodes=0, num_prefills=1,
            source_slots=[0], dest_slots=[7])
        _write_final_states(pool, final, m, 0, transpose_state=False)
        torch.testing.assert_close(pool[7], final[0])

    def test_invalid_dest_slot_skipped(self):
        """DEST slot = -1 → no write."""
        pool = torch.zeros(POOL_SIZE, H, K, V)
        pool_orig = pool.clone()
        final = torch.randn(2, H, K, V)
        m = _make_metadata(
            num_decodes=1, num_prefills=1,
            source_slots=[0, 1], dest_slots=[-1, 3])
        _write_final_states(pool, final, m, 1, transpose_state=False)
        # Decode dest=-1: pool unchanged at all decode slots
        assert (pool[0] == pool_orig[0]).all()
        # Prefill dest=3: written
        torch.testing.assert_close(pool[3], final[1])

    def test_mixed_batch_writes(self):
        """Mixed batch: 2 decodes + 2 prefills all write correctly."""
        pool = torch.zeros(POOL_SIZE, H, V, K)
        final = torch.randn(4, H, K, V)
        m = _make_metadata(
            num_decodes=2, num_prefills=2,
            source_slots=[0, 1, 2, 3], dest_slots=[10, 11, 12, 13])
        _write_final_states(pool, final, m, 2, transpose_state=True)
        for i in range(4):
            expected = final[i].transpose(-1, -2).contiguous()
            torch.testing.assert_close(pool[10 + i], expected)


# ══════════════════════════════════════════════════════════════════
# Tests: _scatter_intermediate_states
# ══════════════════════════════════════════════════════════════════


class TestScatterIntermediateStates:

    def test_single_prefill_two_blocks(self):
        """1 prefill, 128 tokens, block_size=64 → 2 blocks, scatter 1 boundary."""
        pool = torch.zeros(POOL_SIZE, H, K, V)
        # chunk_size=64, 128 tokens → 2 chunks
        # chunk_history shape: [total_chunks, H, K, V]
        # With 0 decodes, prefill_chunk_start=0
        total_chunks = 2
        chunk_history = torch.randn(total_chunks, H, K, V)

        block_table = torch.tensor([[10, 11, -1, -1]], dtype=torch.int32)
        m = _make_metadata(
            num_decodes=0, num_prefills=1,
            source_slots=[-1], dest_slots=[11],
            first_sched=[0], last_sched=[1],  # blocks 0..1, scatter block 0
            num_computed=[0],
            block_table=block_table,
            prefill_chunk_offsets=[0, 2],
            prefill_chunk_start=0,
        )

        _scatter_intermediate_states(
            pool, chunk_history, m, 0, transpose_state=False)

        # n_blocks = last(1) - first(0) = 1
        # chunks_per_block = 64/64 = 1
        # first_aligned_chunk = 0 + 1 = 1 (h[1] = state after block 0)
        # states = chunk_history[1:1+1*1:1] = chunk_history[1]
        # Written to block_table[0, 0] = slot 10
        torch.testing.assert_close(pool[10], chunk_history[1])

    def test_single_prefill_three_blocks_aligned(self):
        """192 tokens, 3 blocks, scatter 2 boundaries (blocks 0, 1)."""
        pool = torch.zeros(POOL_SIZE, H, K, V)
        total_chunks = 3  # 192/64
        chunk_history = torch.randn(total_chunks, H, K, V)

        block_table = torch.tensor([[5, 6, 7, -1]], dtype=torch.int32)
        m = _make_metadata(
            num_decodes=0, num_prefills=1,
            source_slots=[-1], dest_slots=[7],
            first_sched=[0], last_sched=[2],  # scatter blocks 0, 1
            num_computed=[0],
            block_table=block_table,
            prefill_chunk_offsets=[0, 3],
            prefill_chunk_start=0,
        )

        _scatter_intermediate_states(
            pool, chunk_history, m, 0, transpose_state=False)

        # n_blocks=2, first_aligned_chunk=1 (h[k] = state after chunks 0..k-1)
        # states = chunk_history[1:3:1] = [chunk_history[1], chunk_history[2]]
        torch.testing.assert_close(pool[5], chunk_history[1])
        torch.testing.assert_close(pool[6], chunk_history[2])

    def test_with_decode_offset(self):
        """1 decode + 1 prefill, prefill_chunk_start accounts for decode."""
        pool = torch.zeros(POOL_SIZE, H, K, V)
        # 1 decode (1 chunk) + 1 prefill (2 chunks) = 3 total chunks
        total_chunks = 3
        chunk_history = torch.randn(total_chunks, H, K, V)

        block_table = torch.tensor(
            [[8, -1, -1, -1],   # decode
             [9, 10, -1, -1]],  # prefill
            dtype=torch.int32)
        m = _make_metadata(
            num_decodes=1, num_prefills=1,
            source_slots=[8, -1], dest_slots=[8, 10],
            first_sched=[0, 0], last_sched=[0, 1],
            num_computed=[63, 0],  # decode has context, prefill fresh
            block_table=block_table,
            prefill_chunk_offsets=[0, 2],
            prefill_chunk_start=1,  # 1 decode chunk before prefills
        )

        _scatter_intermediate_states(
            pool, chunk_history, m, 1, transpose_state=False)

        # Prefill: n_blocks = 1-0 = 1
        # first_chunk = 1 + 0 = 1 (prefill_chunk_start + offset)
        # first_aligned_chunk = 1 + 1 = 2 (h[2] = state after block 0)
        # num_computed=0, no shift
        # states = chunk_history[2:2+1:1] = chunk_history[2]
        # Written to block_table[1(prefill row 0), 0] = slot 9
        torch.testing.assert_close(pool[9], chunk_history[2])

    def test_no_blocks_to_scatter(self):
        """When first==last, n_blocks=0, nothing scattered."""
        pool = torch.zeros(POOL_SIZE, H, K, V)
        pool_orig = pool.clone()
        chunk_history = torch.randn(2, H, K, V)

        block_table = torch.tensor([[5, -1, -1, -1]], dtype=torch.int32)
        m = _make_metadata(
            num_decodes=0, num_prefills=1,
            source_slots=[-1], dest_slots=[5],
            first_sched=[0], last_sched=[0],  # same → 0 blocks
            num_computed=[0],
            block_table=block_table,
            prefill_chunk_offsets=[0, 2],
            prefill_chunk_start=0,
        )

        _scatter_intermediate_states(
            pool, chunk_history, m, 0, transpose_state=False)
        # Pool unchanged
        torch.testing.assert_close(pool, pool_orig)

    def test_scatter_with_transpose(self):
        """Scatter with transpose [K,V]->[V,K] (Qwen3Next)."""
        pool = torch.zeros(POOL_SIZE, H, V, K)
        chunk_history = torch.randn(2, H, K, V)

        block_table = torch.tensor([[3, 4, -1, -1]], dtype=torch.int32)
        m = _make_metadata(
            num_decodes=0, num_prefills=1,
            source_slots=[-1], dest_slots=[4],
            first_sched=[0], last_sched=[1],
            num_computed=[0],
            block_table=block_table,
            prefill_chunk_offsets=[0, 2],
            prefill_chunk_start=0,
        )

        _scatter_intermediate_states(
            pool, chunk_history, m, 0, transpose_state=True)

        expected = chunk_history[1].transpose(-1, -2).contiguous()
        torch.testing.assert_close(pool[3], expected)

    def test_none_prefill_offsets(self):
        """If prefill_chunk_offsets is None, function returns early."""
        pool = torch.zeros(POOL_SIZE, H, K, V)
        pool_orig = pool.clone()
        chunk_history = torch.randn(1, H, K, V)
        m = _make_metadata(
            num_decodes=1, num_prefills=0,
            source_slots=[0], dest_slots=[1])
        _scatter_intermediate_states(pool, chunk_history, m, 1)
        torch.testing.assert_close(pool, pool_orig)

    def test_resumed_prefill_with_computed_block(self):
        """Resumed prefill: num_computed=64 (one full block cached)."""
        pool = torch.zeros(POOL_SIZE, H, K, V)
        # 64 computed + 96 new = 160 total → blocks 0(cached), 1, 2
        # New tokens: 96 → 2 chunks (96/64 = 1.5 → ceil=2)
        total_chunks = 2
        chunk_history = torch.randn(total_chunks, H, K, V)

        block_table = torch.tensor([[5, 6, 7, -1]], dtype=torch.int32)
        m = _make_metadata(
            num_decodes=0, num_prefills=1,
            source_slots=[5], dest_slots=[7],
            first_sched=[1], last_sched=[2],  # scatter block 1 only
            num_computed=[64],
            block_table=block_table,
            prefill_chunk_offsets=[0, 2],
            prefill_chunk_start=0,
        )

        _scatter_intermediate_states(
            pool, chunk_history, m, 0, transpose_state=False)

        # chunks_per_block=1, first_chunk=0, first_aligned_chunk=0+1=1
        # n_blocks = 2-1 = 1
        # num_computed=64, 64%64=0 → block-aligned, no shift
        # states = chunk_history[1:2:1] = chunk_history[1]
        # h[1] = state after chunk 0 = state after block 1
        torch.testing.assert_close(pool[6], chunk_history[1])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
