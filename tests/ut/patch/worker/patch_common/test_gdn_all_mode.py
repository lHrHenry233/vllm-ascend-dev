# SPDX-License-Identifier: Apache-2.0
"""Unit tests for GDN all-mode prefix caching.

Tests cover:
- _tensor_cdiv: ceiling division for tensors
- _compute_all_mode_block_indices: block index computation
- _apply_all_mode_metadata: metadata attachment and state index override
- _build_initial_state, _write_final_states, _scatter_intermediate_states
"""

import math
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

import vllm_ascend.patch.worker.patch_gdn_attn as patch_gdn_attn
from vllm_ascend.patch.worker.patch_gdn_attn import (
    _compute_all_mode_block_indices,
    _tensor_cdiv,
)
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import MambaSpec


# ---------------------------------------------------------------------------
# Helpers (reuse patterns from existing test_patch_gdn_attn.py)
# ---------------------------------------------------------------------------

@dataclass
class BatchSpec:
    seq_lens: list[int]
    query_lens: list[int]
    name: str = "unnamed"

    @property
    def batch_size(self):
        return len(self.seq_lens)


def create_common_attn_metadata(
    batch_spec: BatchSpec,
    block_size: int,
    device: torch.device = torch.device("cpu"),
) -> CommonAttentionMetadata:
    """Build a CommonAttentionMetadata for testing (same pattern as existing UT)."""
    query_start_loc = torch.zeros(
        batch_spec.batch_size + 1, dtype=torch.int32, device=device
    )
    query_start_loc[1:] = torch.tensor(
        batch_spec.query_lens, dtype=torch.int32, device=device
    ).cumsum(0)
    query_start_loc_cpu = query_start_loc.cpu()
    num_tokens = sum(batch_spec.query_lens)
    seq_lens = torch.tensor(batch_spec.seq_lens, dtype=torch.int32, device=device)
    seq_lens_cpu = seq_lens.cpu()
    max_seq_len = int(seq_lens_cpu.max())
    context_lens = [
        batch_spec.seq_lens[i] - batch_spec.query_lens[i]
        for i in range(batch_spec.batch_size)
    ]
    num_computed_tokens_cpu = torch.tensor(context_lens, dtype=torch.int32)
    max_blocks = (max(batch_spec.seq_lens) + block_size - 1) // block_size
    block_table_tensor = torch.arange(
        batch_spec.batch_size * max_blocks, dtype=torch.int32, device=device
    ).view(batch_spec.batch_size, max_blocks)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)

    return CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens,
        _seq_lens_cpu=seq_lens_cpu,
        _num_computed_tokens_cpu=num_computed_tokens_cpu,
        num_reqs=batch_spec.batch_size,
        num_actual_tokens=num_tokens,
        max_query_len=max(batch_spec.query_lens),
        max_seq_len=max_seq_len,
        block_table_tensor=block_table_tensor,
        slot_mapping=slot_mapping,
        causal=True,
    )


# ---------------------------------------------------------------------------
# _tensor_cdiv tests
# ---------------------------------------------------------------------------

class TestTensorCdiv:
    def test_basic(self):
        a = torch.tensor([0, 1, 4, 5, 7, 8])
        result = _tensor_cdiv(a, 4)
        expected = torch.tensor([0, 1, 1, 2, 2, 2])
        assert torch.equal(result, expected)

    def test_matches_math_ceil(self):
        for n in range(0, 33):
            for d in [1, 2, 4, 7, 16]:
                a = torch.tensor([n])
                got = _tensor_cdiv(a, d).item()
                want = math.ceil(n / d)
                assert got == want, f"cdiv({n}, {d}): got {got}, want {want}"


# ---------------------------------------------------------------------------
# _compute_all_mode_block_indices tests
# ---------------------------------------------------------------------------

class TestComputeAllModeBlockIndices:
    @pytest.mark.parametrize(
        "seq_lens, query_lens, block_size",
        [
            # Fresh prefill: no computed tokens
            ([10], [10], 4),
            # Partial cache hit: 8 tokens computed, 2 new
            ([10], [2], 4),
            # Multiple sequences with different states
            ([10, 20, 4], [2, 5, 4], 4),
            # Exact block boundary
            ([8], [4], 4),
            # Single token
            ([1], [1], 4),
        ],
    )
    def test_block_indices_correctness(self, seq_lens, query_lens, block_size):
        spec = BatchSpec(seq_lens=seq_lens, query_lens=query_lens)
        meta = create_common_attn_metadata(spec, block_size)
        (last_computed, first_sched, last_sched) = _compute_all_mode_block_indices(
            meta, block_size
        )

        for i in range(len(seq_lens)):
            num_computed = seq_lens[i] - query_lens[i]
            expected_lc = max(0, math.ceil(num_computed / block_size) - 1)
            assert last_computed[i].item() == expected_lc, (
                f"seq {i}: last_computed got {last_computed[i].item()}, "
                f"want {expected_lc} (num_computed={num_computed})"
            )
            expected_fs = math.ceil((num_computed + 1) / block_size) - 1
            assert first_sched[i].item() == expected_fs, (
                f"seq {i}: first_sched got {first_sched[i].item()}, "
                f"want {expected_fs}"
            )
            expected_ls = max(0, math.ceil(seq_lens[i] / block_size) - 1)
            assert last_sched[i].item() == expected_ls, (
                f"seq {i}: last_sched got {last_sched[i].item()}, "
                f"want {expected_ls}"
            )

    def test_fresh_prefill_first_block_is_zero(self):
        """When num_computed=0, first_scheduled should be block 0."""
        spec = BatchSpec(seq_lens=[10], query_lens=[10])
        meta = create_common_attn_metadata(spec, block_size=4)
        (last_computed, first_sched, _) = _compute_all_mode_block_indices(meta, 4)
        assert last_computed[0].item() == 0
        assert first_sched[0].item() == 0

    def test_boundary_crossing(self):
        """When num_computed lands exactly on a block boundary."""
        # num_computed=8, block_size=4 → last_computed=1, first_sched=2
        spec = BatchSpec(seq_lens=[12], query_lens=[4])
        meta = create_common_attn_metadata(spec, block_size=4)
        (last_computed, first_sched, last_sched) = _compute_all_mode_block_indices(
            meta, 4
        )
        assert last_computed[0].item() == 1
        assert first_sched[0].item() == 2
        assert last_sched[0].item() == 2


# ---------------------------------------------------------------------------
# _apply_all_mode_metadata tests
# ---------------------------------------------------------------------------

class TestApplyAllModeMetadata:
    def _make_attn_metadata(self, num_decodes, num_prefills):
        return SimpleNamespace(
            num_decodes=num_decodes,
            num_prefills=num_prefills,
        )

    def _make_builder(self, block_size=4):
        spec = MambaSpec(
            block_size=block_size,
            shapes=((1,), (1,)),
            dtypes=(torch.float32,),
            mamba_cache_mode="all",
        )
        builder = SimpleNamespace(kv_cache_spec=spec)
        return builder

    def test_pure_decode_state_indices(self):
        """Decode-only: non_spec_state_indices_tensor points to last block."""
        block_size = 4
        # 2 seqs: seq_lens=[9, 5], query_lens=[1, 1] → all decode
        spec = BatchSpec(seq_lens=[9, 5], query_lens=[1, 1])
        meta = create_common_attn_metadata(spec, block_size)
        attn_metadata = self._make_attn_metadata(num_decodes=2, num_prefills=0)
        builder = self._make_builder(block_size)

        patch_gdn_attn._apply_all_mode_metadata(
            attn_metadata, builder, meta, None
        )

        assert attn_metadata.is_all_mode is True
        assert attn_metadata.mamba_block_size == block_size
        # last_scheduled for seq 0: cdiv(9,4)-1 = 2
        # last_scheduled for seq 1: cdiv(5,4)-1 = 1
        bt = meta.block_table_tensor
        expected_0 = bt[0, 2].item()
        expected_1 = bt[1, 1].item()
        got = attn_metadata.non_spec_state_indices_tensor
        assert got[0].item() == expected_0
        assert got[1].item() == expected_1

    def test_mixed_batch_fields(self):
        """Mixed batch: batch-wide fields cover decode + prefill."""
        block_size = 4
        # 1 decode + 1 prefill
        spec = BatchSpec(seq_lens=[5, 10], query_lens=[1, 6])
        meta = create_common_attn_metadata(spec, block_size)
        attn_metadata = self._make_attn_metadata(num_decodes=1, num_prefills=1)
        builder = self._make_builder(block_size)

        patch_gdn_attn._apply_all_mode_metadata(
            attn_metadata, builder, meta, None
        )

        # Batch-wide fields have size 2 (1 decode + 1 prefill)
        assert attn_metadata.block_state_indices.size(0) == 2
        assert attn_metadata.block_idx_first_scheduled_token.size(0) == 2
        assert attn_metadata.block_idx_last_scheduled_token.size(0) == 2
        assert attn_metadata.block_table_2d.size(0) == 2
        # Forward slices: [:1] = decode, [1:] = prefill
        assert attn_metadata.block_state_indices[:1].size(0) == 1
        assert attn_metadata.block_state_indices[1:].size(0) == 1

    def test_decode_only_batch_fields(self):
        """Decode-only: batch-wide fields have decode entries only."""
        block_size = 4
        spec = BatchSpec(seq_lens=[9], query_lens=[1])
        meta = create_common_attn_metadata(spec, block_size)
        attn_metadata = self._make_attn_metadata(num_decodes=1, num_prefills=0)
        builder = self._make_builder(block_size)

        patch_gdn_attn._apply_all_mode_metadata(
            attn_metadata, builder, meta, None
        )

        assert attn_metadata.block_state_indices.size(0) == 1
        assert attn_metadata.block_idx_first_scheduled_token.size(0) == 1
        assert attn_metadata.block_table_2d.size(0) == 1


# ---------------------------------------------------------------------------
# v3 SSM helper function tests (_build_initial_state, _write_final_states,
# _scatter_intermediate_states)
# ---------------------------------------------------------------------------

class _FakeMetadata:
    """Minimal metadata stub for v3 SSM helper tests."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class TestBuildInitialState:
    def test_reads_from_source_slots(self):
        """Reads initial state from SOURCE pool slots (block_state_indices)."""
        from vllm_ascend.ops.gdn import _build_initial_state

        H, K, V = 2, 4, 4
        ssm_state = torch.randn(8, H, K, V)
        meta = _FakeMetadata(
            block_state_indices=torch.tensor([3, 5], dtype=torch.int32),
            has_initial_state=torch.tensor([True, True]),
        )
        result = _build_initial_state(ssm_state, meta, num_decodes=1,
                                      num_prefills=1, transpose_state=False)
        assert result.shape == (2, H, K, V)
        assert torch.allclose(result[0], ssm_state[3])
        assert torch.allclose(result[1], ssm_state[5])

    def test_transpose_state(self):
        """With transpose_state=True, pool [H,V,K] → kernel [H,K,V]."""
        from vllm_ascend.ops.gdn import _build_initial_state

        H, V, K = 2, 4, 3
        # Pool layout: [N, H, V, K]
        ssm_state = torch.randn(4, H, V, K)
        meta = _FakeMetadata(
            block_state_indices=torch.tensor([1], dtype=torch.int32),
            has_initial_state=torch.tensor([True]),
        )
        result = _build_initial_state(ssm_state, meta, num_decodes=0,
                                      num_prefills=1, transpose_state=True)
        assert result.shape == (1, H, K, V)
        expected = ssm_state[1].transpose(-1, -2)
        assert torch.allclose(result[0], expected)

    def test_zeroes_without_initial_state(self):
        """Sequences with has_initial_state=False get zeroed out."""
        from vllm_ascend.ops.gdn import _build_initial_state

        H, K, V = 2, 4, 4
        ssm_state = torch.randn(4, H, K, V)
        meta = _FakeMetadata(
            block_state_indices=torch.tensor([0, 1], dtype=torch.int32),
            has_initial_state=torch.tensor([True, False]),
        )
        result = _build_initial_state(ssm_state, meta, num_decodes=1,
                                      num_prefills=1, transpose_state=False)
        assert torch.allclose(result[0], ssm_state[0])
        assert torch.all(result[1] == 0)

    def test_invalid_slot_stays_zero(self):
        """SOURCE slot = -1 (no cache) → state stays zero."""
        from vllm_ascend.ops.gdn import _build_initial_state

        H, K, V = 2, 4, 4
        ssm_state = torch.randn(4, H, K, V)
        meta = _FakeMetadata(
            block_state_indices=torch.tensor([-1, 2], dtype=torch.int32),
            has_initial_state=torch.tensor([False, True]),
        )
        result = _build_initial_state(ssm_state, meta, num_decodes=0,
                                      num_prefills=2, transpose_state=False)
        assert torch.all(result[0] == 0)
        assert torch.allclose(result[1], ssm_state[2])


class TestWriteFinalStates:
    def test_writes_to_dest_slots(self):
        """Final state written to DEST pool slots."""
        from vllm_ascend.ops.gdn import _write_final_states

        H, K, V = 2, 4, 4
        ssm_state = torch.zeros(8, H, K, V)
        final_state = torch.randn(2, H, K, V)
        meta = _FakeMetadata(
            non_spec_state_indices_tensor=torch.tensor([3, 7], dtype=torch.int32),
        )
        _write_final_states(ssm_state, final_state, meta, transpose_state=False)
        assert torch.allclose(ssm_state[3], final_state[0])
        assert torch.allclose(ssm_state[7], final_state[1])
        # Other slots untouched
        assert torch.all(ssm_state[0] == 0)

    def test_transpose_on_write(self):
        """With transpose_state=True, kernel [H,K,V] → pool [H,V,K]."""
        from vllm_ascend.ops.gdn import _write_final_states

        H, K, V = 2, 3, 4
        ssm_state = torch.zeros(4, H, V, K)  # Pool layout [H, V, K]
        final_state = torch.randn(1, H, K, V)  # Kernel layout [H, K, V]
        meta = _FakeMetadata(
            non_spec_state_indices_tensor=torch.tensor([2], dtype=torch.int32),
        )
        _write_final_states(ssm_state, final_state, meta, transpose_state=True)
        expected = final_state[0].transpose(-1, -2)
        assert torch.allclose(ssm_state[2], expected)


class TestScatterIntermediateStates:
    def test_single_prefill_two_blocks(self):
        """Single prefill spanning 3 blocks → scatter 2 intermediate states."""
        from vllm_ascend.ops.gdn import _scatter_intermediate_states

        block_size = 8
        chunk_size = 4
        H, K, V = 2, 4, 4

        # 3 blocks, 6 chunks total (8/4=2 chunks per block)
        total_chunks = 6
        intermediate_states = torch.randn(total_chunks, H, K, V)
        ssm_state = torch.zeros(16, H, K, V)  # Large enough for slots 10-12

        meta = _FakeMetadata(
            mamba_block_size=block_size,
            all_mode_chunk_size=chunk_size,
            prefill_chunk_start=0,
            prefill_chunk_offsets=torch.tensor([0, 6], dtype=torch.int32),
            block_table_2d=torch.tensor([[10, 11, 12]], dtype=torch.int32),
            block_idx_first_scheduled_token=torch.tensor([0]),
            block_idx_last_scheduled_token=torch.tensor([2]),
            num_computed_tokens_all=torch.tensor([0], dtype=torch.int32),
        )

        _scatter_intermediate_states(
            ssm_state, intermediate_states, meta,
            num_decodes=0, transpose_state=False,
        )

        # Block 0 ends at chunk 1 (0-indexed), block 1 ends at chunk 3.
        # Block 2 is last → handled by _write_final_states, not scattered.
        assert torch.allclose(ssm_state[10], intermediate_states[1])
        assert torch.allclose(ssm_state[11], intermediate_states[3])
        assert torch.all(ssm_state[12] == 0)  # Last block not scattered

    def test_with_decode_offset(self):
        """Mixed batch: decode chunks precede prefill chunks in h tensor."""
        from vllm_ascend.ops.gdn import _scatter_intermediate_states

        block_size = 4
        chunk_size = 4
        H, K, V = 2, 3, 3

        # 1 decode chunk + 4 prefill chunks (2 blocks → scatter 1)
        total_chunks = 5  # 1 decode + 4 prefill
        intermediate_states = torch.randn(total_chunks, H, K, V)
        ssm_state = torch.zeros(4, H, K, V)

        meta = _FakeMetadata(
            mamba_block_size=block_size,
            all_mode_chunk_size=chunk_size,
            prefill_chunk_start=1,  # decode contributes 1 chunk
            prefill_chunk_offsets=torch.tensor([0, 4], dtype=torch.int32),
            # Row 0 = decode (ignored by scatter), Row 1 = prefill
            block_table_2d=torch.tensor([[99, -1], [0, 1]], dtype=torch.int32),
            block_idx_first_scheduled_token=torch.tensor([99, 0]),
            block_idx_last_scheduled_token=torch.tensor([99, 1]),
            num_computed_tokens_all=torch.tensor([0, 0], dtype=torch.int32),
        )

        _scatter_intermediate_states(
            ssm_state, intermediate_states, meta,
            num_decodes=1, transpose_state=False,
        )

        # Prefill seq: first_sched=0, last_sched=1, n_blocks=1
        # Block 0 ends at chunk: prefill_chunk_start(1) + offset(0) + (0+1)*1 - 1 = 1
        assert torch.allclose(ssm_state[0], intermediate_states[1])
        assert torch.all(ssm_state[1] == 0)  # Last block not scattered

    def test_no_intermediate_blocks(self):
        """Single-block prefill → no scatter (only final state)."""
        from vllm_ascend.ops.gdn import _scatter_intermediate_states

        block_size = 8
        chunk_size = 4
        H, K, V = 2, 4, 4

        intermediate_states = torch.randn(2, H, K, V)
        ssm_state = torch.zeros(4, H, K, V)

        meta = _FakeMetadata(
            mamba_block_size=block_size,
            all_mode_chunk_size=chunk_size,
            prefill_chunk_start=0,
            prefill_chunk_offsets=torch.tensor([0, 2], dtype=torch.int32),
            block_table_2d=torch.tensor([[0]], dtype=torch.int32),
            block_idx_first_scheduled_token=torch.tensor([0]),
            block_idx_last_scheduled_token=torch.tensor([0]),
            num_computed_tokens_all=torch.tensor([0], dtype=torch.int32),
        )

        _scatter_intermediate_states(
            ssm_state, intermediate_states, meta,
            num_decodes=0, transpose_state=False,
        )

        # first_sched=0, last_sched=0, n_blocks=0 → nothing scattered
        assert torch.all(ssm_state == 0)

    def test_transpose_on_scatter(self):
        """With transpose_state=True, kernel [H,K,V] → pool [H,V,K]."""
        from vllm_ascend.ops.gdn import _scatter_intermediate_states

        block_size = 4
        chunk_size = 4
        H, K, V = 2, 3, 5

        total_chunks = 2
        intermediate_states = torch.randn(total_chunks, H, K, V)
        ssm_state = torch.zeros(4, H, V, K)  # Pool layout [H, V, K]

        meta = _FakeMetadata(
            mamba_block_size=block_size,
            all_mode_chunk_size=chunk_size,
            prefill_chunk_start=0,
            prefill_chunk_offsets=torch.tensor([0, 2], dtype=torch.int32),
            block_table_2d=torch.tensor([[0, 1]], dtype=torch.int32),
            block_idx_first_scheduled_token=torch.tensor([0]),
            block_idx_last_scheduled_token=torch.tensor([1]),
            num_computed_tokens_all=torch.tensor([0], dtype=torch.int32),
        )

        _scatter_intermediate_states(
            ssm_state, intermediate_states, meta,
            num_decodes=0, transpose_state=True,
        )

        # Block 0 ends at chunk 0 (cpb=1, index=(0+1)*1-1=0)
        expected = intermediate_states[0].transpose(-1, -2)
        assert torch.allclose(ssm_state[0], expected)
        assert torch.all(ssm_state[1] == 0)  # Last block


# ---------------------------------------------------------------------------
# Additional v3 edge-case and integration tests
# ---------------------------------------------------------------------------

class TestBuildInitialStateExtended:
    """Extended tests for _build_initial_state covering mixed batches and isolation."""

    def test_mixed_batch_multiple_decode_prefill(self):
        """2 decodes + 2 prefills with mixed validity patterns."""
        from vllm_ascend.ops.gdn import _build_initial_state

        H, K, V = 2, 4, 4
        ssm_state = torch.randn(10, H, K, V)
        meta = _FakeMetadata(
            # decode[0]=slot3(valid), decode[1]=slot-1(invalid),
            # prefill[0]=slot7(valid but no initial), prefill[1]=slot2(valid)
            block_state_indices=torch.tensor([3, -1, 7, 2], dtype=torch.int32),
            has_initial_state=torch.tensor([True, False, False, True]),
        )
        result = _build_initial_state(ssm_state, meta, num_decodes=2,
                                      num_prefills=2, transpose_state=False)
        assert result.shape == (4, H, K, V)
        # decode[0]: valid slot 3, has_initial=True → ssm_state[3]
        assert torch.allclose(result[0], ssm_state[3])
        # decode[1]: invalid slot -1 → zero, has_initial=False → stays zero
        assert torch.all(result[1] == 0)
        # prefill[0]: valid slot 7, but has_initial=False → zeroed
        assert torch.all(result[2] == 0)
        # prefill[1]: valid slot 2, has_initial=True → ssm_state[2]
        assert torch.allclose(result[3], ssm_state[2])

    def test_source_dest_isolation(self):
        """Ensure _build_initial_state only reads block_state_indices (SOURCE),
        not non_spec_state_indices_tensor (DEST)."""
        from vllm_ascend.ops.gdn import _build_initial_state

        H, K, V = 2, 3, 3
        ssm_state = torch.randn(8, H, K, V)
        source_val = torch.tensor([1], dtype=torch.int32)
        dest_val = torch.tensor([5], dtype=torch.int32)
        meta = _FakeMetadata(
            block_state_indices=source_val,
            non_spec_state_indices_tensor=dest_val,  # Should be ignored
            has_initial_state=torch.tensor([True]),
        )
        result = _build_initial_state(ssm_state, meta, num_decodes=0,
                                      num_prefills=1, transpose_state=False)
        # Must read from SOURCE slot 1, NOT dest slot 5
        assert torch.allclose(result[0], ssm_state[1])

    def test_all_invalid_slots(self):
        """All slots invalid (-1) → entire initial state is zero."""
        from vllm_ascend.ops.gdn import _build_initial_state

        H, K, V = 2, 4, 4
        ssm_state = torch.randn(4, H, K, V)
        meta = _FakeMetadata(
            block_state_indices=torch.tensor([-1, -1], dtype=torch.int32),
            has_initial_state=torch.tensor([False, False]),
        )
        result = _build_initial_state(ssm_state, meta, num_decodes=1,
                                      num_prefills=1, transpose_state=False)
        assert torch.all(result == 0)


class TestWriteFinalStatesExtended:
    """Extended tests for _write_final_states."""

    def test_dtype_casting(self):
        """Kernel outputs float32 but pool is bfloat16 → cast on write."""
        from vllm_ascend.ops.gdn import _write_final_states

        H, K, V = 2, 4, 4
        ssm_state = torch.zeros(4, H, K, V, dtype=torch.bfloat16)
        final_state = torch.randn(1, H, K, V, dtype=torch.float32)
        meta = _FakeMetadata(
            non_spec_state_indices_tensor=torch.tensor([2], dtype=torch.int32),
        )
        _write_final_states(ssm_state, final_state, meta, transpose_state=False)
        assert ssm_state.dtype == torch.bfloat16
        expected = final_state[0].to(torch.bfloat16)
        assert torch.allclose(ssm_state[2], expected)

    def test_dest_source_isolation(self):
        """Ensure _write_final_states only writes to non_spec_state_indices_tensor (DEST)."""
        from vllm_ascend.ops.gdn import _write_final_states

        H, K, V = 2, 3, 3
        ssm_state = torch.zeros(8, H, K, V)
        final_state = torch.randn(1, H, K, V)
        source_slot = 1
        dest_slot = 5
        meta = _FakeMetadata(
            block_state_indices=torch.tensor([source_slot], dtype=torch.int32),
            non_spec_state_indices_tensor=torch.tensor([dest_slot], dtype=torch.int32),
        )
        _write_final_states(ssm_state, final_state, meta, transpose_state=False)
        # DEST slot 5 should be written
        assert torch.allclose(ssm_state[dest_slot], final_state[0])
        # SOURCE slot 1 must be untouched (still zero)
        assert torch.all(ssm_state[source_slot] == 0)

    def test_multiple_sequences(self):
        """Write final states for 3 sequences to different dest slots."""
        from vllm_ascend.ops.gdn import _write_final_states

        H, K, V = 2, 4, 4
        ssm_state = torch.zeros(10, H, K, V)
        final_state = torch.randn(3, H, K, V)
        meta = _FakeMetadata(
            non_spec_state_indices_tensor=torch.tensor([1, 5, 9], dtype=torch.int32),
        )
        _write_final_states(ssm_state, final_state, meta, transpose_state=False)
        assert torch.allclose(ssm_state[1], final_state[0])
        assert torch.allclose(ssm_state[5], final_state[1])
        assert torch.allclose(ssm_state[9], final_state[2])
        # Other slots untouched
        assert torch.all(ssm_state[0] == 0)
        assert torch.all(ssm_state[3] == 0)


class TestScatterIntermediateStatesExtended:
    """Extended tests for _scatter_intermediate_states."""

    def test_multiple_prefills_different_blocks(self):
        """3 prefills: 1-block, 2-block, 3-block → scatter 0, 1, 2 intermediate states."""
        from vllm_ascend.ops.gdn import _scatter_intermediate_states

        block_size = 4
        chunk_size = 4  # cpb = 1
        H, K, V = 2, 3, 3

        # Prefill 0: 1 block → 1 chunk, no scatter
        # Prefill 1: 2 blocks → 2 chunks, scatter block 0
        # Prefill 2: 3 blocks → 3 chunks, scatter blocks 0 and 1
        total_chunks = 1 + 2 + 3  # = 6
        intermediate_states = torch.randn(total_chunks, H, K, V)
        ssm_state = torch.zeros(20, H, K, V)

        meta = _FakeMetadata(
            mamba_block_size=block_size,
            all_mode_chunk_size=chunk_size,
            prefill_chunk_start=0,
            prefill_chunk_offsets=torch.tensor([0, 1, 3, 6], dtype=torch.int32),
            block_table_2d=torch.tensor([
                [10, -1, -1],  # prefill 0: 1 block
                [11, 12, -1],  # prefill 1: 2 blocks
                [13, 14, 15],  # prefill 2: 3 blocks
            ], dtype=torch.int32),
            block_idx_first_scheduled_token=torch.tensor([0, 0, 0]),
            block_idx_last_scheduled_token=torch.tensor([0, 1, 2]),
            num_computed_tokens_all=torch.tensor([0, 0, 0], dtype=torch.int32),
        )

        _scatter_intermediate_states(
            ssm_state, intermediate_states, meta,
            num_decodes=0, transpose_state=False,
        )

        # Prefill 0: last_sched=0, first_sched=0, n_blocks=0 → no scatter
        assert torch.all(ssm_state[10] == 0)

        # Prefill 1: n_blocks=1, block 0 at chunk index seq_start(1) + (0+1)*1 - 1 = 1
        assert torch.allclose(ssm_state[11], intermediate_states[1])
        assert torch.all(ssm_state[12] == 0)  # Last block

        # Prefill 2: n_blocks=2, block 0 at chunk 3+(0+1)*1-1=3, block 1 at 3+(1+1)*1-1=4
        assert torch.allclose(ssm_state[13], intermediate_states[3])
        assert torch.allclose(ssm_state[14], intermediate_states[4])
        assert torch.all(ssm_state[15] == 0)  # Last block

    def test_large_chunks_per_block(self):
        """block_size=128, chunk_size=64 → cpb=2. Stride-select every 2nd chunk."""
        from vllm_ascend.ops.gdn import _scatter_intermediate_states

        block_size = 128
        chunk_size = 64
        H, K, V = 2, 4, 4

        # 3 blocks → 6 chunks (cpb=2)
        total_chunks = 6
        intermediate_states = torch.randn(total_chunks, H, K, V)
        ssm_state = torch.zeros(10, H, K, V)

        meta = _FakeMetadata(
            mamba_block_size=block_size,
            all_mode_chunk_size=chunk_size,
            prefill_chunk_start=0,
            prefill_chunk_offsets=torch.tensor([0, 6], dtype=torch.int32),
            block_table_2d=torch.tensor([[0, 1, 2]], dtype=torch.int32),
            block_idx_first_scheduled_token=torch.tensor([0]),
            block_idx_last_scheduled_token=torch.tensor([2]),
            num_computed_tokens_all=torch.tensor([0], dtype=torch.int32),
        )

        _scatter_intermediate_states(
            ssm_state, intermediate_states, meta,
            num_decodes=0, transpose_state=False,
        )

        # cpb=2, block 0 ends at chunk (0+1)*2-1=1, block 1 at (1+1)*2-1=3
        assert torch.allclose(ssm_state[0], intermediate_states[1])
        assert torch.allclose(ssm_state[1], intermediate_states[3])
        assert torch.all(ssm_state[2] == 0)  # Last block

    def test_invalid_cache_slot_skipped(self):
        """block_table contains -1 (padding) → that slot should be skipped."""
        from vllm_ascend.ops.gdn import _scatter_intermediate_states

        block_size = 4
        chunk_size = 4
        H, K, V = 2, 3, 3

        # 3 blocks, but middle block's cache slot = -1
        total_chunks = 3
        intermediate_states = torch.randn(total_chunks, H, K, V)
        ssm_state = torch.zeros(8, H, K, V)

        meta = _FakeMetadata(
            mamba_block_size=block_size,
            all_mode_chunk_size=chunk_size,
            prefill_chunk_start=0,
            prefill_chunk_offsets=torch.tensor([0, 3], dtype=torch.int32),
            block_table_2d=torch.tensor([[5, -1, 7]], dtype=torch.int32),
            block_idx_first_scheduled_token=torch.tensor([0]),
            block_idx_last_scheduled_token=torch.tensor([2]),
            num_computed_tokens_all=torch.tensor([0], dtype=torch.int32),
        )

        _scatter_intermediate_states(
            ssm_state, intermediate_states, meta,
            num_decodes=0, transpose_state=False,
        )

        # Block 0 (slot=5): valid → scatter
        assert torch.allclose(ssm_state[5], intermediate_states[0])
        # Block 1 (slot=-1): invalid → skip
        # Block 2: last block → handled by _write_final_states
        assert torch.all(ssm_state[7] == 0)


class TestAllModePipelineIntegration:
    """End-to-end integration test: build → mock kernel → write_final → scatter.

    Uses known synthetic data to verify the complete all-mode SSM state flow
    without requiring actual Triton kernel execution.
    """

    def test_full_pipeline_mixed_batch(self):
        """1 decode + 1 prefill (3 blocks). Verify:
        - Initial state reads from SOURCE slots
        - Final state writes to DEST slots
        - Intermediate states scatter to correct pool positions
        - SOURCE ≠ DEST, no cross-contamination
        """
        from vllm_ascend.ops.gdn import (
            _build_initial_state,
            _write_final_states,
            _scatter_intermediate_states,
        )

        H, K, V = 2, 4, 4
        block_size = 8
        chunk_size = 4  # cpb = 2

        # Pool has 16 slots
        ssm_state = torch.zeros(16, H, K, V)
        # Pre-populate SOURCE slots with known values
        source_decode_slot = 3
        source_prefill_slot = 7
        ssm_state[source_decode_slot] = torch.full((H, K, V), 1.0)
        ssm_state[source_prefill_slot] = torch.full((H, K, V), 2.0)

        dest_decode_slot = 5
        dest_prefill_slot = 12  # last-scheduled block slot

        meta = _FakeMetadata(
            # SOURCE: where to read initial state
            block_state_indices=torch.tensor(
                [source_decode_slot, source_prefill_slot], dtype=torch.int32
            ),
            has_initial_state=torch.tensor([True, True]),
            # DEST: where to write final state
            non_spec_state_indices_tensor=torch.tensor(
                [dest_decode_slot, dest_prefill_slot], dtype=torch.int32
            ),
            # Scatter metadata (prefill only)
            mamba_block_size=block_size,
            all_mode_chunk_size=chunk_size,
            prefill_chunk_start=1,  # 1 decode chunk before prefill
            prefill_chunk_offsets=torch.tensor([0, 6], dtype=torch.int32),
            block_table_2d=torch.tensor([
                [99, -1, -1],  # decode row (ignored by scatter)
                [10, 11, 12],  # prefill: 3 blocks
            ], dtype=torch.int32),
            block_idx_first_scheduled_token=torch.tensor([99, 0]),
            block_idx_last_scheduled_token=torch.tensor([99, 2]),
            num_computed_tokens_all=torch.tensor([0, 0], dtype=torch.int32),
        )

        # Step 1: Build initial state
        initial_state = _build_initial_state(
            ssm_state, meta, num_decodes=1, num_prefills=1,
            transpose_state=False,
        )
        assert initial_state.shape == (2, H, K, V)
        assert torch.allclose(initial_state[0], torch.full((H, K, V), 1.0))
        assert torch.allclose(initial_state[1], torch.full((H, K, V), 2.0))

        # Step 2: Mock kernel output
        # In reality, kernel returns (o, final_state, intermediate_states)
        # We mock: final_state for both seqs, intermediate_states for all chunks
        final_state = torch.randn(2, H, K, V)
        # 1 decode chunk + 6 prefill chunks = 7 total
        intermediate_states = torch.randn(7, H, K, V)

        # Step 3: Write final states to DEST
        _write_final_states(ssm_state, final_state, meta, transpose_state=False)
        assert torch.allclose(ssm_state[dest_decode_slot], final_state[0])
        assert torch.allclose(ssm_state[dest_prefill_slot], final_state[1])
        # SOURCE slots should NOT be overwritten (they're different from DEST)
        assert torch.allclose(
            ssm_state[source_decode_slot], torch.full((H, K, V), 1.0)
        )

        # Step 4: Scatter intermediate states (prefill only)
        _scatter_intermediate_states(
            ssm_state, intermediate_states, meta,
            num_decodes=1, transpose_state=False,
        )
        # Prefill: 3 blocks, cpb=2, scatter blocks 0 and 1 (block 2 = last → skip)
        # Block 0: chunk index = prefill_start(1) + offset(0) + (0+1)*2 - 1 = 2
        # Block 1: chunk index = 1 + 0 + (1+1)*2 - 1 = 4
        assert torch.allclose(ssm_state[10], intermediate_states[2])
        assert torch.allclose(ssm_state[11], intermediate_states[4])
        # SOURCE slots still intact
        assert torch.allclose(
            ssm_state[source_decode_slot], torch.full((H, K, V), 1.0)
        )
        assert torch.allclose(
            ssm_state[source_prefill_slot], torch.full((H, K, V), 2.0)
        )

    def test_pipeline_with_transpose(self):
        """Same pipeline but with transpose_state=True (Qwen3Next layout)."""
        from vllm_ascend.ops.gdn import (
            _build_initial_state,
            _write_final_states,
            _scatter_intermediate_states,
        )

        H, V, K = 2, 5, 3  # Pool: [H, V, K], Kernel: [H, K, V]
        block_size = 4
        chunk_size = 4  # cpb = 1

        ssm_state = torch.zeros(8, H, V, K)  # Pool layout
        # Pre-populate source with known pool-layout data
        source_slot = 1
        pool_data = torch.randn(H, V, K)
        ssm_state[source_slot] = pool_data

        dest_slot = 6

        meta = _FakeMetadata(
            block_state_indices=torch.tensor([source_slot], dtype=torch.int32),
            has_initial_state=torch.tensor([True]),
            non_spec_state_indices_tensor=torch.tensor([dest_slot], dtype=torch.int32),
            mamba_block_size=block_size,
            all_mode_chunk_size=chunk_size,
            prefill_chunk_start=0,
            prefill_chunk_offsets=torch.tensor([0, 2], dtype=torch.int32),
            block_table_2d=torch.tensor([[3, 4]], dtype=torch.int32),
            block_idx_first_scheduled_token=torch.tensor([0]),
            block_idx_last_scheduled_token=torch.tensor([1]),
            num_computed_tokens_all=torch.tensor([0], dtype=torch.int32),
        )

        # Build: pool [H,V,K] → kernel [H,K,V]
        initial_state = _build_initial_state(
            ssm_state, meta, num_decodes=0, num_prefills=1,
            transpose_state=True,
        )
        assert initial_state.shape == (1, H, K, V)
        assert torch.allclose(initial_state[0], pool_data.transpose(-1, -2))

        # Mock kernel output in kernel layout [H, K, V]
        final_state = torch.randn(1, H, K, V)
        intermediate_states = torch.randn(2, H, K, V)

        # Write final: kernel [H,K,V] → pool [H,V,K]
        _write_final_states(ssm_state, final_state, meta, transpose_state=True)
        expected_pool = final_state[0].transpose(-1, -2)
        assert torch.allclose(ssm_state[dest_slot], expected_pool)

        # Scatter: kernel [H,K,V] → pool [H,V,K]
        _scatter_intermediate_states(
            ssm_state, intermediate_states, meta,
            num_decodes=0, transpose_state=True,
        )
        # Block 0 at chunk 0 (cpb=1), scattered to slot 3
        expected_scatter = intermediate_states[0].transpose(-1, -2)
        assert torch.allclose(ssm_state[3], expected_scatter)
        assert torch.all(ssm_state[4] == 0)  # Last block not scattered
