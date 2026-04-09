# SPDX-License-Identifier: Apache-2.0
"""Unit tests for GDN all-mode prefix caching.

Tests cover:
- _tensor_cdiv: ceiling division for tensors
- _compute_all_mode_block_indices: block index computation
- _apply_all_mode_metadata: metadata attachment and state index override
- _write_intermediate_conv_states: conv state writeback at block boundaries
- _prefill_ssm_all_mode: multi-pass SSM processing
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
from vllm_ascend.ops.gdn import (
    _write_intermediate_conv_states,
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

    def test_prefill_sets_p_fields(self):
        """Prefill: _p suffix fields are set for prefill-only subset."""
        block_size = 4
        # 1 decode + 1 prefill
        spec = BatchSpec(seq_lens=[5, 10], query_lens=[1, 6])
        meta = create_common_attn_metadata(spec, block_size)
        attn_metadata = self._make_attn_metadata(num_decodes=1, num_prefills=1)
        builder = self._make_builder(block_size)

        patch_gdn_attn._apply_all_mode_metadata(
            attn_metadata, builder, meta, None
        )

        # _p fields should only have 1 element (the prefill seq)
        assert attn_metadata.block_idx_first_scheduled_token_p.size(0) == 1
        assert attn_metadata.block_idx_last_computed_token_p.size(0) == 1
        assert attn_metadata.block_table_2d_p.size(0) == 1

    def test_decode_only_no_p_fields(self):
        """Decode-only: _p fields should be None."""
        block_size = 4
        spec = BatchSpec(seq_lens=[9], query_lens=[1])
        meta = create_common_attn_metadata(spec, block_size)
        attn_metadata = self._make_attn_metadata(num_decodes=1, num_prefills=0)
        builder = self._make_builder(block_size)

        patch_gdn_attn._apply_all_mode_metadata(
            attn_metadata, builder, meta, None
        )

        assert attn_metadata.block_idx_first_scheduled_token_p is None
        assert attn_metadata.block_table_2d_p is None


# ---------------------------------------------------------------------------
# _write_intermediate_conv_states tests
# ---------------------------------------------------------------------------

class TestWriteIntermediateConvStates:
    def test_single_seq_two_blocks(self):
        """Prefill spans 2 blocks → should write conv state at block 0 boundary."""
        block_size = 4
        conv_width = 3  # state_width = conv_width - 1 = 2
        dim = 8

        # Sequence: 7 tokens, 0 computed → blocks [0, 1]
        # Block 0 ends at position 4
        num_tokens = 7
        mixed_qkv = torch.randn(num_tokens, dim)
        # Cache: 4 pages (more than needed)
        conv_state_cache = torch.zeros(4, dim, conv_width - 1)  # SD layout
        block_table_2d = torch.tensor([[0, 1]], dtype=torch.int32)
        block_idx_first_sched = torch.tensor([0], dtype=torch.int32)
        block_idx_last_sched = torch.tensor([1], dtype=torch.int32)
        block_idx_last_computed = torch.tensor([0], dtype=torch.int32)
        num_computed = torch.tensor([0], dtype=torch.int32)
        query_start_loc = torch.tensor([0, 7], dtype=torch.int32)

        _write_intermediate_conv_states(
            mixed_qkv, conv_state_cache, block_table_2d,
            block_idx_first_sched, block_idx_last_sched,
            block_idx_last_computed, num_computed, block_size,
            query_start_loc, conv_width,
        )

        # Block 0 boundary at position 4: conv state = input[2:4] (last 2 values before pos 4)
        expected_state = mixed_qkv[2:4].T  # shape [dim, 2]
        assert torch.allclose(conv_state_cache[0], expected_state), (
            f"got {conv_state_cache[0]}, expected {expected_state}"
        )
        # Block 1 (last block) should NOT be written by this function
        assert torch.all(conv_state_cache[1] == 0)

    def test_first_boundary_loads_from_cache(self):
        """When first boundary needs values before scheduled range, load from cache."""
        block_size = 4
        conv_width = 3  # state_width = 2
        dim = 4

        # Sequence: seq_len=9, num_computed=3, query_len=6
        # Scheduled tokens cover positions 3..8
        # Block 0 ends at position 4, needs positions [2, 3] for conv state
        # Position 2 is BEFORE scheduled range → must come from cache
        num_tokens = 6
        mixed_qkv = torch.randn(num_tokens, dim)

        # Pre-populate cache at block_idx_last_computed (block 0, slot 0)
        # with some known values
        conv_state_cache = torch.zeros(4, dim, conv_width - 1)  # SD: [pages, dim, state_len]
        initial_conv_state = torch.randn(dim, conv_width - 1)
        conv_state_cache[0] = initial_conv_state  # slot 0 = block 0

        block_table_2d = torch.tensor([[0, 1, 2]], dtype=torch.int32)
        block_idx_first_sched = torch.tensor([0], dtype=torch.int32)  # cdiv(3+1, 4)-1 = 0
        block_idx_last_sched = torch.tensor([2], dtype=torch.int32)  # cdiv(9, 4)-1 = 2
        block_idx_last_computed = torch.tensor([0], dtype=torch.int32)  # cdiv(3, 4)-1 = 0
        num_computed = torch.tensor([3], dtype=torch.int32)
        query_start_loc = torch.tensor([0, 6], dtype=torch.int32)

        _write_intermediate_conv_states(
            mixed_qkv, conv_state_cache, block_table_2d,
            block_idx_first_sched, block_idx_last_sched,
            block_idx_last_computed, num_computed, block_size,
            query_start_loc, conv_width,
        )

        # Block 0 boundary at abs pos 4 → needs [pos 2, pos 3]
        # pos 2 is before scheduled range (num_computed=3, scheduled starts at 3)
        # → 1 value from cache, 1 from input
        # cache_part = initial_conv_state.T[-1:] (last 1 row of transposed)
        cache_part = initial_conv_state.T[-1:]  # shape [1, dim]
        input_part = mixed_qkv[0:1]  # position 3 = scheduled[0], shape [1, dim]
        expected_state = torch.cat([cache_part, input_part], dim=0).T  # [dim, 2]
        assert torch.allclose(conv_state_cache[0], expected_state, atol=1e-6), (
            f"Cache mismatch at block 0"
        )

    def test_no_intermediate_blocks(self):
        """When scheduled range fits in one block, nothing should be written."""
        block_size = 4
        conv_width = 3
        dim = 4
        num_tokens = 3
        mixed_qkv = torch.randn(num_tokens, dim)
        conv_state_cache = torch.zeros(2, dim, conv_width - 1)
        block_table_2d = torch.tensor([[0, 1]], dtype=torch.int32)
        block_idx_first_sched = torch.tensor([1], dtype=torch.int32)
        block_idx_last_sched = torch.tensor([1], dtype=torch.int32)
        block_idx_last_computed = torch.tensor([0], dtype=torch.int32)
        num_computed = torch.tensor([4], dtype=torch.int32)
        query_start_loc = torch.tensor([0, 3], dtype=torch.int32)

        _write_intermediate_conv_states(
            mixed_qkv, conv_state_cache, block_table_2d,
            block_idx_first_sched, block_idx_last_sched,
            block_idx_last_computed, num_computed, block_size,
            query_start_loc, conv_width,
        )
        # No blocks should be modified
        assert torch.all(conv_state_cache == 0)


# ---------------------------------------------------------------------------
# _prefill_ssm_all_mode tests (mock kernel)
# ---------------------------------------------------------------------------

class TestPrefillSsmAllMode:
    def test_single_seq_single_block(self):
        """Single sequence fitting in one block → 1 pass, state written to cache."""
        from vllm_ascend.ops.gdn import _prefill_ssm_all_mode

        block_size = 8
        num_heads = 2
        head_dim_k = 4
        head_dim_v = 4
        num_tokens = 5

        # Create inputs: [1, T, H, D]
        q = torch.randn(1, num_tokens, num_heads, head_dim_k)
        k = torch.randn(1, num_tokens, num_heads, head_dim_k)
        v = torch.randn(1, num_tokens, num_heads, head_dim_v)
        g = torch.randn(1, num_tokens, num_heads, head_dim_k)
        beta = torch.randn(1, num_tokens, num_heads)

        # SSM state cache: [num_pages, H, V, K]
        ssm_state = torch.zeros(2, num_heads, head_dim_v, head_dim_k)
        block_table_2d = torch.tensor([[0, 1]], dtype=torch.int32)

        call_count = [0]
        received_states = []

        def mock_kernel(*, q, k, v, g, beta, initial_state,
                        output_final_state, cu_seqlens, head_first,
                        use_qk_l2norm_in_kernel):
            call_count[0] += 1
            received_states.append(initial_state.clone())
            T = q.size(1)
            out = torch.randn_like(v)
            new_state = torch.randn(1, num_heads, head_dim_k, head_dim_v)
            return out, new_state

        output = _prefill_ssm_all_mode(
            chunk_gated_delta_rule_fn=mock_kernel,
            q=q, k=k, v=v, g=g, beta=beta,
            ssm_state=ssm_state,
            block_table_2d=block_table_2d,
            block_idx_first_scheduled_token=torch.tensor([0]),
            block_idx_last_scheduled_token=torch.tensor([0]),
            block_idx_last_computed_token=torch.tensor([0]),
            num_computed_tokens=torch.tensor([0]),
            block_size=block_size,
            non_spec_query_start_loc=torch.tensor([0, num_tokens]),
            has_initial_state=torch.tensor([False]),
            head_first=False,
            use_qk_l2norm_in_kernel=True,
            transpose_state=False,
        )

        assert call_count[0] == 1, f"Expected 1 kernel call, got {call_count[0]}"
        assert output.size(1) == num_tokens
        # SSM state at block 0 should be updated
        assert not torch.all(ssm_state[0] == 0), "State should be written"

    def test_multi_block_multi_pass(self):
        """Sequence spanning 3 blocks → 3 passes."""
        from vllm_ascend.ops.gdn import _prefill_ssm_all_mode

        block_size = 4
        num_heads = 2
        head_dim = 4
        num_tokens = 10  # 3 blocks: [0..3], [4..7], [8..9]

        q = torch.randn(1, num_tokens, num_heads, head_dim)
        k = torch.randn(1, num_tokens, num_heads, head_dim)
        v = torch.randn(1, num_tokens, num_heads, head_dim)
        g = torch.randn(1, num_tokens, num_heads, head_dim)
        beta = torch.randn(1, num_tokens, num_heads)

        ssm_state = torch.zeros(4, num_heads, head_dim, head_dim)
        block_table_2d = torch.tensor([[0, 1, 2]], dtype=torch.int32)

        call_count = [0]

        def mock_kernel(*, q, k, v, g, beta, initial_state,
                        output_final_state, cu_seqlens, head_first,
                        use_qk_l2norm_in_kernel):
            call_count[0] += 1
            batch = initial_state.size(0)
            out = torch.randn(1, q.size(1), num_heads, head_dim)
            new_state = torch.randn(batch, num_heads, head_dim, head_dim)
            return out, new_state

        output = _prefill_ssm_all_mode(
            chunk_gated_delta_rule_fn=mock_kernel,
            q=q, k=k, v=v, g=g, beta=beta,
            ssm_state=ssm_state,
            block_table_2d=block_table_2d,
            block_idx_first_scheduled_token=torch.tensor([0]),
            block_idx_last_scheduled_token=torch.tensor([2]),
            block_idx_last_computed_token=torch.tensor([0]),
            num_computed_tokens=torch.tensor([0]),
            block_size=block_size,
            non_spec_query_start_loc=torch.tensor([0, num_tokens]),
            has_initial_state=torch.tensor([False]),
            head_first=False,
            use_qk_l2norm_in_kernel=True,
            transpose_state=False,
        )

        assert call_count[0] == 3, f"Expected 3 passes, got {call_count[0]}"
        assert output.size(1) == num_tokens
        # All 3 blocks should have state written
        for blk in range(3):
            assert not torch.all(ssm_state[blk] == 0), f"Block {blk} state not written"
