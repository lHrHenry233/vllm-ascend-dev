# SPDX-License-Identifier: Apache-2.0
"""Unit tests for _compute_all_mode_metadata in patch_gdn_attn.py.

Tests verify SOURCE/DEST pool slot computation, block index derivation,
and chunk offset calculation for all-mode prefix caching metadata.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.patch.worker.patch_gdn_attn import _compute_all_mode_metadata

BLOCK_SIZE = 64
CHUNK_SIZE = 64  # _GDN_CHUNK_SIZE


def _make_builder(block_size: int = BLOCK_SIZE):
    """Create a minimal builder with kv_cache_spec."""
    return SimpleNamespace(
        kv_cache_spec=SimpleNamespace(block_size=block_size),
    )


def _make_attn_metadata(num_decodes: int, num_prefills: int):
    """Create a minimal attn_metadata object."""
    return SimpleNamespace(
        num_decodes=num_decodes,
        num_prefills=num_prefills,
        spec_state_indices_tensor=None,
        non_spec_state_indices_tensor=None,
    )


def _make_common_attn_metadata(
    seq_lens: list[int],
    query_lens: list[int],
    block_table: torch.Tensor | None = None,
    block_size: int = BLOCK_SIZE,
    device: torch.device = torch.device("cpu"),
):
    """Create CommonAttentionMetadata-like object.

    Args:
        seq_lens: Per-seq total length (after this step).
        query_lens: Per-seq query length (decode=1, prefill>1).
        block_table: [batch, max_blocks] pool slot IDs. If None, auto-generated.
        block_size: Block size for auto-generating block_table.
        device: Tensor device.
    """
    batch_size = len(seq_lens)
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    query_start_loc = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    query_start_loc[1:] = torch.tensor(
        query_lens, dtype=torch.int32, device=device
    ).cumsum(0)

    if block_table is None:
        max_blocks = max((s + block_size - 1) // block_size for s in seq_lens)
        # Pool slot IDs: seq_i block_j → i*100 + j (easy to verify)
        block_table = torch.full(
            (batch_size, max_blocks), -1, dtype=torch.int32, device=device
        )
        for i, s in enumerate(seq_lens):
            n_blocks = (s + block_size - 1) // block_size
            for j in range(n_blocks):
                block_table[i, j] = i * 100 + j

    return SimpleNamespace(
        seq_lens=seq_lens_t,
        query_start_loc=query_start_loc,
        block_table_tensor=block_table,
    )


class TestAllModeDecodeOnly:
    """Decode-only batch: all seqs have query_len=1."""

    def test_single_seq_mid_block(self):
        """Seq at 100 tokens → block 1. New token at 101 → still block 1."""
        builder = _make_builder()
        attn = _make_attn_metadata(num_decodes=1, num_prefills=0)
        m = _make_common_attn_metadata(
            seq_lens=[101], query_lens=[1],
        )
        _compute_all_mode_metadata(builder, attn, m)

        assert attn.is_all_mode is True
        # context_len=100, source block=(100-1)//64=1, pool slot=0*100+1=1
        assert attn.block_state_indices.tolist() == [1]
        # seq_len=101, dest block=(101-1)//64=1, pool slot=0*100+1=1
        assert attn.non_spec_state_indices_tensor.tolist() == [1]
        # SOURCE == DEST (same block)
        assert attn.block_idx_first_scheduled_token.tolist() == [100 // 64]  # 1
        assert attn.block_idx_last_scheduled_token.tolist() == [100 // 64]  # 1
        assert attn.num_computed_tokens_all.tolist() == [100]
        assert attn.prefill_chunk_offsets is None

    def test_block_boundary_crossing(self):
        """Seq at 64 tokens → block 0 full. New token at 65 → block 1."""
        builder = _make_builder()
        attn = _make_attn_metadata(num_decodes=1, num_prefills=0)
        m = _make_common_attn_metadata(
            seq_lens=[65], query_lens=[1],
        )
        _compute_all_mode_metadata(builder, attn, m)

        # context_len=64, source block=(64-1)//64=0, pool slot=0
        assert attn.block_state_indices.tolist() == [0]
        # seq_len=65, dest block=(65-1)//64=1, pool slot=1
        assert attn.non_spec_state_indices_tensor.tolist() == [1]
        # SOURCE ≠ DEST (crosses block boundary)
        assert attn.block_idx_first_scheduled_token.tolist() == [64 // 64]  # 1
        assert attn.block_idx_last_scheduled_token.tolist() == [1]

    def test_multi_seq_batch(self):
        """Multiple decode seqs with different positions."""
        builder = _make_builder()
        attn = _make_attn_metadata(num_decodes=3, num_prefills=0)
        m = _make_common_attn_metadata(
            seq_lens=[10, 65, 200], query_lens=[1, 1, 1],
        )
        _compute_all_mode_metadata(builder, attn, m)

        # seq0: ctx=9, src_blk=0, slot=0*100+0=0; dest_blk=0, slot=0
        # seq1: ctx=64, src_blk=0, slot=1*100+0=100; dest_blk=1, slot=1*100+1=101
        # seq2: ctx=199, src_blk=3(=199//64), slot=2*100+3=203; dest_blk=3, slot=203
        assert attn.block_state_indices.tolist() == [0, 100, 203]
        assert attn.non_spec_state_indices_tensor.tolist() == [0, 101, 203]
        assert attn.prefill_chunk_start == 3  # 3 decode seqs


class TestAllModePrefillOnly:
    """Prefill-only batch: all seqs have query_len > 1."""

    def test_new_seq_no_context(self):
        """Brand new seq, no prior state. SOURCE should be -1."""
        builder = _make_builder()
        attn = _make_attn_metadata(num_decodes=0, num_prefills=1)
        m = _make_common_attn_metadata(
            seq_lens=[128], query_lens=[128],
        )
        _compute_all_mode_metadata(builder, attn, m)

        # context_len=0 → SOURCE = -1 (no prior state)
        assert attn.block_state_indices.tolist() == [-1]
        # dest block = (128-1)//64 = 1, pool slot = 0*100+1 = 1
        assert attn.non_spec_state_indices_tensor.tolist() == [1]
        assert attn.block_idx_first_scheduled_token.tolist() == [0]
        assert attn.block_idx_last_scheduled_token.tolist() == [1]
        assert attn.num_computed_tokens_all.tolist() == [0]

    def test_continuation_prefill(self):
        """Seq with existing context gets more tokens."""
        builder = _make_builder()
        attn = _make_attn_metadata(num_decodes=0, num_prefills=1)
        # Already has 100 tokens, gets 200 more → total 300
        m = _make_common_attn_metadata(
            seq_lens=[300], query_lens=[200],
        )
        _compute_all_mode_metadata(builder, attn, m)

        # context=100, src_blk=(100-1)//64=1, slot=0*100+1=1
        assert attn.block_state_indices.tolist() == [1]
        # dest_blk=(300-1)//64=4, slot=0*100+4=4
        assert attn.non_spec_state_indices_tensor.tolist() == [4]
        assert attn.block_idx_first_scheduled_token.tolist() == [100 // 64]  # 1
        assert attn.block_idx_last_scheduled_token.tolist() == [4]

    def test_chunk_offsets(self):
        """Verify prefill_chunk_offsets computation."""
        builder = _make_builder()
        attn = _make_attn_metadata(num_decodes=0, num_prefills=3)
        # 3 prefills with different query lengths
        m = _make_common_attn_metadata(
            seq_lens=[64, 200, 30],
            query_lens=[64, 200, 30],  # all new seqs
        )
        _compute_all_mode_metadata(builder, attn, m)

        # chunks per seq: ceil(64/64)=1, ceil(200/64)=4, ceil(30/64)=1
        assert attn.prefill_chunk_offsets.tolist() == [0, 1, 5, 6]
        assert attn.prefill_chunk_start == 0  # no decodes


class TestAllModeMixedBatch:
    """Mixed batch: decode seqs first, then prefill seqs."""

    def test_mixed_decode_prefill(self):
        """2 decodes + 1 prefill."""
        builder = _make_builder()
        attn = _make_attn_metadata(num_decodes=2, num_prefills=1)
        m = _make_common_attn_metadata(
            seq_lens=[50, 130, 256],
            query_lens=[1, 1, 128],
        )
        _compute_all_mode_metadata(builder, attn, m)

        # Decode seq0: ctx=49, src_blk=0, slot=0; dest_blk=0, slot=0
        # Decode seq1: ctx=129, src_blk=2(=129//64), slot=100+2=102; dest_blk=2, slot=102
        # Prefill seq2: ctx=128, src_blk=1(=127//64), slot=200+1=201; dest_blk=3, slot=200+3=203
        assert attn.block_state_indices.tolist() == [0, 102, 201]
        assert attn.non_spec_state_indices_tensor.tolist() == [0, 102, 203]
        assert attn.num_computed_tokens_all.tolist() == [49, 129, 128]

        # Prefill chunks
        assert attn.prefill_chunk_start == 2  # 2 decode seqs
        # Prefill seq2: 128 tokens → ceil(128/64)=2 chunks
        assert attn.prefill_chunk_offsets.tolist() == [0, 2]

    def test_block_idx_ranges(self):
        """Verify first/last scheduled block indices for scatter."""
        builder = _make_builder()
        attn = _make_attn_metadata(num_decodes=1, num_prefills=1)
        # Decode: ctx=63→block0, seq_len=64→block0 (still same block)
        # Prefill: ctx=64→block1, seq_len=256→block3 (spans blocks 1-3)
        m = _make_common_attn_metadata(
            seq_lens=[64, 256],
            query_lens=[1, 192],
        )
        _compute_all_mode_metadata(builder, attn, m)

        # first_scheduled = context_len // block_size
        assert attn.block_idx_first_scheduled_token.tolist() == [63 // 64, 64 // 64]
        # [0, 1]
        # last_scheduled = (seq_len - 1) // block_size
        assert attn.block_idx_last_scheduled_token.tolist() == [63 // 64, 255 // 64]
        # [0, 3]


class TestAllModeEdgeCases:
    """Edge cases and boundary conditions."""

    def test_exactly_on_block_boundary(self):
        """context_len is exact multiple of block_size."""
        builder = _make_builder()
        attn = _make_attn_metadata(num_decodes=0, num_prefills=1)
        # ctx=128 (exactly 2 blocks), prefill 64 more tokens
        m = _make_common_attn_metadata(
            seq_lens=[192], query_lens=[64],
        )
        _compute_all_mode_metadata(builder, attn, m)

        # src_blk = (128-1)//64 = 1 (last token of block 1)
        assert attn.block_state_indices.tolist() == [1]
        # first_scheduled = 128//64 = 2 (new tokens start at block 2)
        assert attn.block_idx_first_scheduled_token.tolist() == [2]
        # dest_blk = (192-1)//64 = 2
        assert attn.block_idx_last_scheduled_token.tolist() == [2]

    def test_single_token_seq(self):
        """Seq with just 1 token total (new, no context)."""
        builder = _make_builder()
        attn = _make_attn_metadata(num_decodes=0, num_prefills=1)
        m = _make_common_attn_metadata(
            seq_lens=[1], query_lens=[1],
        )
        _compute_all_mode_metadata(builder, attn, m)

        # No context → SOURCE = -1
        assert attn.block_state_indices.tolist() == [-1]
        # dest = block 0
        assert attn.non_spec_state_indices_tensor.tolist() == [0]
        assert attn.block_idx_first_scheduled_token.tolist() == [0]
        assert attn.block_idx_last_scheduled_token.tolist() == [0]

    def test_small_block_size(self):
        """Use block_size=16 to test with more blocks."""
        builder = _make_builder(block_size=16)
        attn = _make_attn_metadata(num_decodes=1, num_prefills=1)
        # Decode: ctx=47→block2, seq_len=48→block2
        # Prefill: ctx=32→block2(src=block1), seq_len=96→block5
        m = _make_common_attn_metadata(
            seq_lens=[48, 96], query_lens=[1, 64], block_size=16,
        )
        _compute_all_mode_metadata(builder, attn, m)

        # seq0: src_blk=(47-1)//16=2, slot=0*100+2=2; dest_blk=(48-1)//16=2, slot=2
        # seq1: src_blk=(32-1)//16=1, slot=1*100+1=101; dest_blk=(96-1)//16=5, slot=1*100+5=105
        assert attn.block_state_indices.tolist() == [2, 101]
        assert attn.non_spec_state_indices_tensor.tolist() == [2, 105]

    def test_spec_decode_raises(self):
        """All-mode with spec decode should raise NotImplementedError."""
        builder = _make_builder()
        attn = _make_attn_metadata(num_decodes=1, num_prefills=0)
        attn.spec_state_indices_tensor = torch.tensor([0])
        m = _make_common_attn_metadata(
            seq_lens=[10], query_lens=[1],
        )
        with pytest.raises(NotImplementedError, match="spec decode"):
            _compute_all_mode_metadata(builder, attn, m)

    def test_metadata_fields_complete(self):
        """Verify all expected fields are set on attn_metadata."""
        builder = _make_builder()
        attn = _make_attn_metadata(num_decodes=1, num_prefills=1)
        m = _make_common_attn_metadata(
            seq_lens=[65, 128], query_lens=[1, 128],
        )
        _compute_all_mode_metadata(builder, attn, m)

        expected_fields = [
            "is_all_mode", "mamba_block_size", "all_mode_chunk_size",
            "block_table_2d", "block_state_indices",
            "block_idx_first_scheduled_token", "block_idx_last_scheduled_token",
            "num_computed_tokens_all", "prefill_chunk_start",
            "prefill_chunk_offsets", "non_spec_state_indices_tensor",
        ]
        for field in expected_fields:
            assert hasattr(attn, field), f"Missing field: {field}"

        assert attn.mamba_block_size == BLOCK_SIZE
        assert attn.all_mode_chunk_size == CHUNK_SIZE
        assert attn.block_table_2d.shape[0] == 2  # num_seqs
