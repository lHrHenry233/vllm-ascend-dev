from types import SimpleNamespace
from unittest.mock import patch

import torch

from tests.ut.base import TestBase
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm_ascend.patch.worker.patch_qwen3_5 import (
    AscendQwen3_5GatedDeltaNet,
    _ensure_prefill_token_state_indices,
    _is_mamba_all_prefix_mode,
    _write_all_mode_token_conv_states,
)


class TestPatchQwen35AllModeHelpers(TestBase):

    def test_is_mamba_all_prefix_mode_true(self):
        forward_context = SimpleNamespace(
            vllm_config=SimpleNamespace(
                cache_config=SimpleNamespace(
                    enable_prefix_caching=True,
                    mamba_cache_mode="all",
                )
            )
        )
        self.assertTrue(_is_mamba_all_prefix_mode(forward_context))

    def test_is_mamba_all_prefix_mode_false_when_not_all(self):
        forward_context = SimpleNamespace(
            vllm_config=SimpleNamespace(
                cache_config=SimpleNamespace(
                    enable_prefix_caching=True,
                    mamba_cache_mode="align",
                )
            )
        )
        self.assertFalse(_is_mamba_all_prefix_mode(forward_context))

    def test_ensure_prefill_token_state_indices_returns_2d_directly(self):
        state_indices = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)
        query_start_loc = torch.tensor([0, 2, 4], dtype=torch.long)
        result = _ensure_prefill_token_state_indices(state_indices, query_start_loc)
        self.assertTrue(torch.equal(result, state_indices))

    def test_ensure_prefill_token_state_indices_expand_with_pad(self):
        state_indices = torch.tensor([7, 9], dtype=torch.long)
        query_start_loc = torch.tensor([0, 2, 5], dtype=torch.long)

        result = _ensure_prefill_token_state_indices(state_indices, query_start_loc)

        expected = torch.tensor(
            [
                [7, 7, PAD_SLOT_ID],
                [9, 9, 9],
            ],
            dtype=torch.long,
        )
        self.assertTrue(torch.equal(result, expected))

    def test_ensure_prefill_token_state_indices_shape_mismatch_fallback(self):
        state_indices = torch.tensor([1, 2, 3], dtype=torch.long)
        query_start_loc = torch.tensor([0, 2, 5], dtype=torch.long)

        result = _ensure_prefill_token_state_indices(state_indices, query_start_loc)

        self.assertEqual(result.shape, (3, 1))
        self.assertTrue(torch.equal(result.squeeze(1), state_indices))

    def test_write_all_mode_token_conv_states_without_initial_state(self):
        mixed_qkv_non_spec = torch.tensor(
            [
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
            ],
            dtype=torch.float32,
        )
        conv_state = torch.zeros((4, 2, 3), dtype=torch.float32)
        token_state_indices = torch.tensor([[1, 2, 3]], dtype=torch.int32)
        query_start_loc = torch.tensor([0, 3], dtype=torch.int32)
        has_initial_state = torch.tensor([False], dtype=torch.bool)

        _write_all_mode_token_conv_states(
            mixed_qkv_non_spec,
            conv_state,
            token_state_indices,
            query_start_loc,
            has_initial_state,
            state_width=3,
        )

        self.assertTrue(torch.equal(conv_state[1, 0, :3], torch.tensor([0.0, 0.0, 1.0])))
        self.assertTrue(torch.equal(conv_state[2, 0, :3], torch.tensor([0.0, 1.0, 2.0])))
        self.assertTrue(torch.equal(conv_state[3, 0, :3], torch.tensor([1.0, 2.0, 3.0])))

    def test_write_all_mode_token_conv_states_with_initial_state_and_pad(self):
        mixed_qkv_non_spec = torch.tensor(
            [
                [5.0, 50.0],
                [6.0, 60.0],
            ],
            dtype=torch.float32,
        )
        conv_state = torch.zeros((5, 2, 3), dtype=torch.float32)
        conv_state[4, :, :3] = torch.tensor([[7.0, 8.0, 9.0], [70.0, 80.0, 90.0]])
        token_state_indices = torch.tensor([[4, PAD_SLOT_ID, 2]], dtype=torch.int32)
        query_start_loc = torch.tensor([0, 2], dtype=torch.int32)
        has_initial_state = torch.tensor([True], dtype=torch.bool)

        _write_all_mode_token_conv_states(
            mixed_qkv_non_spec,
            conv_state,
            token_state_indices,
            query_start_loc,
            has_initial_state,
            state_width=3,
        )

        self.assertTrue(torch.equal(conv_state[4, 0, :3], torch.tensor([8.0, 9.0, 5.0])))
        self.assertTrue(torch.equal(conv_state[2, 0, :3], torch.tensor([0.0, 0.0, 0.0])))

    @patch("vllm_ascend.patch.worker.patch_qwen3_5.maybe_save_kv_layer_to_connector")
    @patch("vllm_ascend.patch.worker.patch_qwen3_5.enable_sp", return_value=False)
    @patch("vllm_ascend.patch.worker.patch_qwen3_5.GDNAttentionMetadata", new=SimpleNamespace)
    @patch("vllm_ascend.patch.worker.patch_qwen3_5.fused_gdn_gating_patch")
    @patch("vllm_ascend.patch.worker.patch_qwen3_5.fused_recurrent_gated_delta_rule")
    @patch("vllm_ascend.patch.worker.patch_qwen3_5.torch.ops._C_ascend.causal_conv1d_fn", create=True)
    @patch("vllm_ascend.patch.worker.patch_qwen3_5.get_forward_context")
    def test_forward_core_all_mode_prefill_uses_token_level_indices(
        self,
        mock_get_forward_context,
        mock_causal_conv1d_fn,
        mock_fused_recurrent,
        mock_fused_gating,
        _mock_enable_sp,
        _mock_save_kv,
    ):
        # Build a minimal forward context for all-mode prefill.
        non_spec_query_start_loc = torch.tensor([0, 3], dtype=torch.int32)
        non_spec_state_indices = torch.tensor([5], dtype=torch.int32)
        attn_metadata = SimpleNamespace(
            has_initial_state=torch.tensor([False], dtype=torch.bool),
            spec_query_start_loc=None,
            non_spec_query_start_loc=non_spec_query_start_loc,
            spec_sequence_masks=None,
            spec_token_indx=None,
            non_spec_token_indx=None,
            spec_state_indices_tensor=None,
            non_spec_state_indices_tensor=non_spec_state_indices,
            num_actual_tokens=3,
            num_accepted_tokens=0,
            num_prefills=1,
            num_decodes=0,
            num_spec_decodes=0,
        )
        forward_context = SimpleNamespace(
            attn_metadata={"pref": attn_metadata},
            virtual_engine=0,
            vllm_config=SimpleNamespace(
                cache_config=SimpleNamespace(
                    enable_prefix_caching=True,
                    mamba_cache_mode="all",
                ))
        )
        mock_get_forward_context.return_value = forward_context

        mock_fused_gating.return_value = (
            torch.zeros((1, 3, 2), dtype=torch.float32),
            torch.zeros((1, 3, 2), dtype=torch.float32),
        )
        mock_fused_recurrent.return_value = (
            torch.zeros((1, 3, 2), dtype=torch.float32),
            torch.zeros((1, 2), dtype=torch.float32),
        )

        # Patch custom op entry to keep test purely CPU/UT level.
        mock_causal_conv1d_fn.side_effect = lambda *args, **kwargs: args[0]

        class FakeSelf:
            pass

        fake_self = FakeSelf()
        fake_self.prefix = "pref"
        fake_self.activation = "silu"
        fake_self.A_log = torch.zeros((2,), dtype=torch.float32)
        fake_self.dt_bias = torch.zeros((2,), dtype=torch.float32)
        fake_self.conv1d = SimpleNamespace(
            weight=torch.zeros((2, 1, 4), dtype=torch.float32),
            bias=torch.zeros((2,), dtype=torch.float32),
        )
        # cache layout: [num_cache_lines, state_len, dim]
        fake_self.kv_cache = [(
            torch.zeros((8, 3, 2), dtype=torch.float32),
            torch.ones((8, 2), dtype=torch.float32),
        )]
        fake_self.rearrange_mixed_qkv = (
            lambda x: (None, None, None) if x is None else (x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
        )

        mixed_qkv = torch.tensor(
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
            dtype=torch.float32,
        )
        b = torch.zeros_like(mixed_qkv)
        a = torch.zeros_like(mixed_qkv)
        core_attn_out = torch.zeros((3, 2), dtype=torch.float32)

        AscendQwen3_5GatedDeltaNet._forward_core(fake_self, mixed_qkv, b, a, core_attn_out)

        self.assertTrue(mock_fused_recurrent.called)
        recurrent_kwargs = mock_fused_recurrent.call_args.kwargs
        token_indices = recurrent_kwargs["ssm_state_indices"]
        self.assertEqual(token_indices.shape, (1, 3))
        self.assertTrue(torch.equal(token_indices, torch.tensor([[5, 5, 5]], dtype=torch.int32)))

    @patch("vllm_ascend.patch.worker.patch_qwen3_5.maybe_save_kv_layer_to_connector")
    @patch("vllm_ascend.patch.worker.patch_qwen3_5.enable_sp", return_value=False)
    @patch("vllm_ascend.patch.worker.patch_qwen3_5.GDNAttentionMetadata", new=SimpleNamespace)
    @patch("vllm_ascend.patch.worker.patch_qwen3_5.fused_sigmoid_gating_delta_rule_update")
    @patch("vllm_ascend.patch.worker.patch_qwen3_5.fused_gdn_gating_patch")
    @patch("vllm_ascend.patch.worker.patch_qwen3_5.fused_recurrent_gated_delta_rule")
    @patch("vllm_ascend.patch.worker.patch_qwen3_5.causal_conv1d_update")
    @patch("vllm_ascend.patch.worker.patch_qwen3_5.get_forward_context")
    def test_forward_core_all_mode_decode_uses_recurrent_path(
        self,
        mock_get_forward_context,
        mock_causal_conv1d_update,
        mock_fused_recurrent,
        mock_fused_gating,
        mock_sigmoid_update,
        _mock_enable_sp,
        _mock_save_kv,
    ):
        non_spec_query_start_loc = torch.tensor([0, 2], dtype=torch.int32)
        non_spec_state_indices = torch.tensor([3, 4], dtype=torch.int32)
        attn_metadata = SimpleNamespace(
            has_initial_state=torch.tensor([True, True], dtype=torch.bool),
            spec_query_start_loc=None,
            non_spec_query_start_loc=non_spec_query_start_loc,
            spec_sequence_masks=None,
            spec_token_indx=None,
            non_spec_token_indx=None,
            spec_state_indices_tensor=None,
            non_spec_state_indices_tensor=non_spec_state_indices,
            num_actual_tokens=2,
            num_accepted_tokens=0,
            num_prefills=0,
            num_decodes=2,
            num_spec_decodes=0,
        )
        forward_context = SimpleNamespace(
            attn_metadata={"pref": attn_metadata},
            virtual_engine=0,
            vllm_config=SimpleNamespace(
                cache_config=SimpleNamespace(
                    enable_prefix_caching=True,
                    mamba_cache_mode="all",
                ))
        )
        mock_get_forward_context.return_value = forward_context

        mock_causal_conv1d_update.side_effect = lambda x, *_args, **_kwargs: x
        mock_fused_gating.return_value = (
            torch.zeros((1, 2, 2), dtype=torch.float32),
            torch.zeros((1, 2, 2), dtype=torch.float32),
        )
        mock_fused_recurrent.return_value = (
            torch.zeros((1, 2, 2), dtype=torch.float32),
            torch.zeros((2, 2), dtype=torch.float32),
        )

        class FakeSelf:
            pass

        fake_self = FakeSelf()
        fake_self.prefix = "pref"
        fake_self.activation = "silu"
        fake_self.A_log = torch.zeros((2,), dtype=torch.float32)
        fake_self.dt_bias = torch.zeros((2,), dtype=torch.float32)
        fake_self.conv1d = SimpleNamespace(
            weight=torch.zeros((2, 1, 4), dtype=torch.float32),
            bias=torch.zeros((2,), dtype=torch.float32),
        )
        fake_self.kv_cache = [(
            torch.zeros((8, 3, 2), dtype=torch.float32),
            torch.ones((8, 2), dtype=torch.float32),
        )]
        fake_self.rearrange_mixed_qkv = (
            lambda x: (None, None, None) if x is None else (x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
        )

        mixed_qkv = torch.tensor(
            [[1.0, 10.0], [2.0, 20.0]],
            dtype=torch.float32,
        )
        b = torch.zeros_like(mixed_qkv)
        a = torch.zeros_like(mixed_qkv)
        core_attn_out = torch.zeros((2, 2), dtype=torch.float32)

        AscendQwen3_5GatedDeltaNet._forward_core(fake_self, mixed_qkv, b, a, core_attn_out)

        self.assertTrue(mock_fused_recurrent.called)
        recurrent_kwargs = mock_fused_recurrent.call_args.kwargs
        self.assertTrue(torch.equal(recurrent_kwargs["ssm_state_indices"], non_spec_state_indices))
        self.assertFalse(mock_sigmoid_update.called)
