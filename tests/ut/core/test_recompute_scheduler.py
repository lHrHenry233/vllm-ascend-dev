from types import SimpleNamespace
from unittest.mock import patch

from tests.ut.base import TestBase
from vllm_ascend.core.recompute_scheduler import RecomputeScheduler


def _fake_scheduler_init(self, *args, **kwargs):
    self.vllm_config = kwargs["vllm_config"]
    # Baseline behavior for hybrid mamba models.
    self.need_mamba_block_aligned_split = True


class TestRecomputeSchedulerAllMode(TestBase):

    def _build_vllm_config(self, model_type: str, mamba_cache_mode: str, enable_prefix_caching: bool = True):
        return SimpleNamespace(
            speculative_config=None,
            kv_transfer_config=None,
            model_config=SimpleNamespace(
                hf_text_config=SimpleNamespace(model_type=model_type),
            ),
            cache_config=SimpleNamespace(
                mamba_cache_mode=mamba_cache_mode,
                enable_prefix_caching=enable_prefix_caching,
            ),
        )

    @patch("vllm_ascend.core.recompute_scheduler.register_ascend_mla_spec_in_manager", lambda: None)
    @patch("vllm_ascend.core.recompute_scheduler.Scheduler.__init__", new=_fake_scheduler_init)
    def test_all_mode_disables_block_aligned_split_for_qwen3_5(self):
        cfg = self._build_vllm_config("qwen3_5", "all")

        scheduler = RecomputeScheduler(vllm_config=cfg)

        self.assertFalse(scheduler.need_mamba_block_aligned_split)

    @patch("vllm_ascend.core.recompute_scheduler.register_ascend_mla_spec_in_manager", lambda: None)
    @patch("vllm_ascend.core.recompute_scheduler.Scheduler.__init__", new=_fake_scheduler_init)
    def test_align_mode_keeps_block_aligned_split_for_qwen3_5(self):
        cfg = self._build_vllm_config("qwen3_5", "align")

        scheduler = RecomputeScheduler(vllm_config=cfg)

        self.assertTrue(scheduler.need_mamba_block_aligned_split)

    @patch("vllm_ascend.core.recompute_scheduler.register_ascend_mla_spec_in_manager", lambda: None)
    @patch("vllm_ascend.core.recompute_scheduler.Scheduler.__init__", new=_fake_scheduler_init)
    def test_all_mode_non_hybrid_keeps_unchanged(self):
        cfg = self._build_vllm_config("llama", "all")

        scheduler = RecomputeScheduler(vllm_config=cfg)

        self.assertTrue(scheduler.need_mamba_block_aligned_split)

    @patch("vllm_ascend.core.recompute_scheduler.register_ascend_mla_spec_in_manager", lambda: None)
    @patch("vllm_ascend.core.recompute_scheduler.Scheduler.__init__", new=_fake_scheduler_init)
    def test_all_mode_hybrid_without_prefix_caching_keeps_unchanged(self):
        cfg = self._build_vllm_config("qwen3_next", "all", enable_prefix_caching=False)

        scheduler = RecomputeScheduler(vllm_config=cfg)

        self.assertTrue(scheduler.need_mamba_block_aligned_split)
