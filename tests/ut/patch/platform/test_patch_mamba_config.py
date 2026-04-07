from types import SimpleNamespace
from unittest.mock import patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.patch.platform.patch_mamba_config import verify_and_update_config


class _FakeModelCls:

    @staticmethod
    def get_mamba_state_shape_from_config(_vllm_config):
        return [(128,), (64,)]

    @staticmethod
    def get_mamba_state_dtype_from_config(_vllm_config):
        return [torch.float16, torch.float16]


class TestPatchMambaConfig(TestBase):

    def _build_vllm_config(self, mamba_cache_mode: str):
        cache_config = SimpleNamespace(
            cache_dtype="auto",
            block_size=16,
            mamba_page_size_padded=None,
            enable_prefix_caching=True,
            mamba_cache_mode=mamba_cache_mode,
            mamba_block_size=None,
        )
        model_config = SimpleNamespace(
            dtype=torch.float16,
            architecture="FakeArch",
            max_model_len=4096,
            get_num_kv_heads=lambda _parallel: 1,
            get_head_size=lambda: 1,
        )
        parallel_config = SimpleNamespace()
        return SimpleNamespace(
            cache_config=cache_config,
            model_config=model_config,
            parallel_config=parallel_config,
        )

    @patch("vllm_ascend.patch.platform.patch_mamba_config.MambaModelConfig.verify_and_update_config")
    @patch("vllm_ascend.patch.platform.patch_mamba_config.ModelRegistry.resolve_model_cls")
    def test_all_mode_uses_block_size(self, mock_resolve, _mock_verify):
        mock_resolve.return_value = (_FakeModelCls, None)
        vllm_config = self._build_vllm_config("all")

        verify_and_update_config.__func__(None, vllm_config)

        self.assertEqual(
            vllm_config.cache_config.mamba_block_size,
            vllm_config.cache_config.block_size,
        )

    @patch("vllm_ascend.patch.platform.patch_mamba_config.MambaModelConfig.verify_and_update_config")
    @patch("vllm_ascend.patch.platform.patch_mamba_config.ModelRegistry.resolve_model_cls")
    def test_align_mode_uses_block_size(self, mock_resolve, _mock_verify):
        mock_resolve.return_value = (_FakeModelCls, None)
        vllm_config = self._build_vllm_config("align")

        verify_and_update_config.__func__(None, vllm_config)

        self.assertEqual(
            vllm_config.cache_config.mamba_block_size,
            vllm_config.cache_config.block_size,
        )

    @patch("vllm_ascend.patch.platform.patch_mamba_config.MambaModelConfig.verify_and_update_config")
    @patch("vllm_ascend.patch.platform.patch_mamba_config.ModelRegistry.resolve_model_cls")
    def test_none_mode_uses_max_model_len(self, mock_resolve, _mock_verify):
        mock_resolve.return_value = (_FakeModelCls, None)
        vllm_config = self._build_vllm_config("none")

        verify_and_update_config.__func__(None, vllm_config)

        self.assertEqual(
            vllm_config.cache_config.mamba_block_size,
            vllm_config.model_config.max_model_len,
        )
