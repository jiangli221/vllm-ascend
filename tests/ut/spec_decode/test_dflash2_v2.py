#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Unit tests for the V2 DFlash2 speculator (worker/v2/spec_decode/dflash2)."""

import sys
from contextlib import nullcontext
from dataclasses import make_dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.forward_context import is_forward_context_available, override_forward_context
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

from vllm_ascend.worker.v2.spec_decode import init_speculator
from vllm_ascend.worker.v2.spec_decode.dflash.speculator import AscendDFlashSpeculator
from vllm_ascend.worker.v2.spec_decode.dflash2.speculator import (
    AscendDFlash2Speculator,
    _selector_walk_kernel_ascend,
)

@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("hardware", ["A2", "A3"])
@pytest.mark.parametrize("target_family", ["qwen"])
@pytest.mark.parametrize("target_kv_dtype", ["auto", "fp8"])
def test_v2_loader_selects_flash_attn_and_restores_target_context(
    monkeypatch, causal, fail, hardware, target_family, target_kv_dtype
):
    from vllm.compilation import backends
    from vllm.config import AttentionConfig, CacheConfig, set_current_vllm_config
    from vllm.v1.attention import selector as selector_module
    from vllm.v1.attention.selector import AttentionSelectorConfig
    from vllm.v1.worker.gpu.spec_decode.dflash import utils as loader

    from vllm_ascend import platform
    from vllm_ascend.device.hardware import AscendDeviceType
    from vllm_ascend.device.hardware_profile import get_hardware_profile

    is_glm = target_family == "glm"
    target_config = SimpleNamespace(
        runner_type="generate",
        architecture="GlmMoeDsaForCausalLM" if is_glm else "Qwen3_5ForConditionalGeneration",
        hf_text_config=SimpleNamespace(model_type="glm_moe_dsa" if is_glm else "qwen3_5_text"),
    )
    draft_config = SimpleNamespace(
        architectures=["DFlash2DraftModel"],
        hf_config=SimpleNamespace(num_hidden_layers=6, is_causal=causal),
    )
    spec = SimpleNamespace(
        method="dflash",
        target_model_config=target_config,
        draft_model_config=draft_config,
        enforce_eager=True,
        attention_backend=None,
        kv_cache_dtype="auto",
    )
    # Keep the real V2 loader's config cloning, causality and model-tag scopes.
    fields = {
        "use_v2_model_runner": True,
        "model_config": target_config,
        "speculative_config": spec,
        "attention_config": AttentionConfig(),
        "cache_config": CacheConfig(cache_dtype=target_kv_dtype),
        "load_config": object(),
        "parallel_config": SimpleNamespace(
            pipeline_parallel_size=1, prefill_context_parallel_size=1, decode_context_parallel_size=1
        ),
        "quant_config": None,
        "kv_transfer_config": None,
    }
    config_cls = make_dataclass("LoaderConfig", [(name, object) for name in fields])
    config = config_cls(**fields)
    draft_model = SimpleNamespace(model=object(), config=draft_config.hf_config)
    target_model = SimpleNamespace(model=object())
    monkeypatch.setattr(loader, "get_draft_load_config", lambda _: config.load_config)
    monkeypatch.setattr(loader, "get_pp_safe_draft_load_config", lambda value: value)
    monkeypatch.setattr(loader, "maybe_share_target_embed", lambda *_: None)
    monkeypatch.setattr(loader, "get_target_lm_head", lambda *_: None)
    monkeypatch.setattr(
        platform,
        "get_current_hardware_profile",
        lambda: get_hardware_profile(AscendDeviceType[hardware]),
    )
    optional_ops = SimpleNamespace(flash_attn_with_kvcache=MagicMock(), get_scheduler_metadata=MagicMock())
    monkeypatch.setitem(sys.modules, "flash_attn_npu_3", optional_ops)
    find_spec = platform.util.find_spec
    monkeypatch.setattr(
        platform.util, "find_spec", lambda name: object() if name == "flash_attn_npu_3" else find_spec(name)
    )
    monkeypatch.setattr(platform, "_validate_fa3_backend", lambda *_: False)
    monkeypatch.setattr("vllm.platforms.current_platform", platform.NPUPlatform)
    monkeypatch.setattr(selector_module, "resolve_obj_by_qualname", lambda path: path)
    select = selector_module._cached_get_attn_backend
    target_selector = AttentionSelectorConfig(
        head_size=256,
        block_size=128,
        dtype=torch.bfloat16,
        kv_cache_dtype=target_kv_dtype,
        use_mla=is_glm,
        use_sparse=is_glm,
    )
    target_backend = (
        "vllm_ascend.attention.sfa_v1.AscendSFABackend"
        if is_glm
        else "vllm_ascend.attention.attention_v1.AscendAttentionBackend"
    )
    if not is_glm and hardware in ("A2", "A3") and target_kv_dtype == "auto":
        target_backend = "vllm_ascend.attention.flash_attention_v3.AscendQwenFlashAttentionBackend"
    if is_glm and hardware == "A5":
        draft_backend = "vllm_ascend.attention.fa_v1.AscendFlashAttentionBackend"
    elif not is_glm and hardware in ("A2", "A3"):
        draft_backend = "vllm_ascend.attention.flash_attention_v3.AscendFlashAttentionBackend"
    else:
        draft_backend = "vllm_ascend.attention.attention_v1.AscendAttentionBackend"
    initial_tag = backends.model_tag

    def get_model(*, vllm_config, model_config):
        assert model_config is draft_config
        assert vllm_config.model_config is target_config
        assert vllm_config.cache_config.cache_dtype == "auto"
        assert vllm_config.attention_config.use_non_causal is (not causal)
        assert backends.model_tag == "dflash_head"
        selector = target_selector._replace(
            head_size=128, use_mla=False, use_sparse=False, kv_cache_dtype="auto", use_non_causal=not causal
        )
        with set_current_vllm_config(vllm_config):
            assert select(None, selector) == draft_backend
        if fail:
            raise RuntimeError("draft load failed")
        return draft_model

    monkeypatch.setattr(loader, "get_model", get_model)
    speculator = AscendDFlash2Speculator.__new__(AscendDFlash2Speculator)
    speculator.vllm_config = config
    select.cache_clear()
    try:
        with set_current_vllm_config(config):
            assert select(None, target_selector) == target_backend
            expected = pytest.raises(RuntimeError, match="draft load failed") if fail else nullcontext()
            with expected:
                assert speculator.load_draft_model(target_model, set()) is draft_model
            assert backends.model_tag == initial_tag
            assert select(None, target_selector) == target_backend
        assert config.model_config is target_config
        assert config.cache_config.cache_dtype == target_kv_dtype
        assert not config.attention_config.use_non_causal
        assert select.cache_info().currsize == 2
    finally:
        select.cache_clear()

def test_dflash_loader_temporarily_disables_profiling_chunk(monkeypatch, fail):
    additional_config = {"profiling_chunk_config": {"enabled": "true", "min_chunk": 64}}
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=2),
        additional_config=additional_config,
    )
    speculator = AscendDFlashSpeculator.__new__(AscendDFlashSpeculator)
    speculator.vllm_config = config
    draft = object()

    def load(self, target_model, target_attn_layer_names):
        del target_model, target_attn_layer_names
        assert self.vllm_config is config
        assert self.vllm_config.additional_config["profiling_chunk_config"]["enabled"] is False
        if fail:
            raise RuntimeError("draft load failed")
        return draft

    monkeypatch.setattr(DFlashSpeculator, "load_draft_model", load)
    expected_context = pytest.raises(RuntimeError, match="draft load failed") if fail else nullcontext()
    with expected_context:
        assert speculator.load_draft_model(object(), set()) is draft

    assert config.additional_config is additional_config
    assert additional_config["profiling_chunk_config"]["enabled"] == "true"


def test_patch_swaps_upstream_selector_walk_kernel():
    import vllm.v1.worker.gpu.spec_decode.dflash2.speculator as upstream

    import vllm_ascend.patch.worker.patch_v2.patch_dflash_speculator  # noqa: F401

    assert upstream._selector_walk_kernel is _selector_walk_kernel_ascend


def _spec_config(arch: str) -> SimpleNamespace:
    return SimpleNamespace(
        method="dflash",
        enforce_eager=None,
        use_dspark=lambda: False,
        use_dflash=lambda: True,
        draft_model_config=SimpleNamespace(architectures=[arch]),
    )


def test_init_speculator_routes_dflash2_draft_model():
    cfg = SimpleNamespace(speculative_config=_spec_config("DFlash2DraftModel"))
    with patch("vllm_ascend.worker.v2.spec_decode.dflash2.speculator.AscendDFlash2Speculator") as d2:
        assert init_speculator(cfg, torch.device("cpu")) is d2.return_value
        d2.assert_called_once_with(cfg, torch.device("cpu"))

    cfg = SimpleNamespace(speculative_config=_spec_config("DFlashDraftModel"))
    with (
        patch("vllm_ascend.worker.v2.spec_decode.dflash2.speculator.AscendDFlash2Speculator") as d2,
        patch("vllm_ascend.worker.v2.spec_decode.dflash.speculator.AscendDFlashSpeculator") as d1,
    ):
        assert init_speculator(cfg, torch.device("cpu")) is d1.return_value
        d2.assert_not_called()


@pytest.mark.parametrize(
    "enforce_eager,expected_mode",
    [(False, CUDAGraphMode.FULL_DECODE_ONLY), (None, CUDAGraphMode.FULL_DECODE_ONLY), (True, CUDAGraphMode.NONE)],
)
def test_init_cudagraph_manager_preserves_graph_mode(monkeypatch, enforce_eager, expected_mode):
    calls = []
    speculator = AscendDFlash2Speculator.__new__(AscendDFlash2Speculator)
    speculator.speculative_config = SimpleNamespace(enforce_eager=enforce_eager)
    speculator.update_stream = object()

    def initialize_parent(self, mode):
        calls.append(mode)
        self.query_cudagraph_manager = SimpleNamespace()

    monkeypatch.setattr(DFlashSpeculator, "init_cudagraph_manager", initialize_parent)
    speculator.init_cudagraph_manager(CUDAGraphMode.FULL_DECODE_ONLY)

    assert calls == [expected_mode]
    assert speculator.query_cudagraph_manager.speculator is speculator
    assert speculator.query_cudagraph_manager.update_stream is speculator.update_stream


def test_selector_scores_without_forward_context():
    speculator = AscendDFlash2Speculator.__new__(AscendDFlash2Speculator)
    speculator.num_query_per_req = 3
    speculator.input_buffers = SimpleNamespace(input_ids=torch.arange(9))
    candidates = torch.zeros(2, 2, 2, dtype=torch.int64)
    unary = torch.zeros(2, 2, 2)
    hidden = torch.zeros(2, 2, 4)
    scores = torch.zeros(2, 2, 2, 2)

    def select(candidate_ids, unary_logits, hidden_states, anchor_token_ids):
        assert not is_forward_context_available()
        assert candidate_ids is candidates
        assert unary_logits is unary
        assert hidden_states is hidden
        torch.testing.assert_close(anchor_token_ids, torch.tensor([0, 3]))
        return scores

    selector = MagicMock(side_effect=select)
    speculator.model = SimpleNamespace(model=SimpleNamespace(candidate_selector=selector))
    with override_forward_context(None):
        assert speculator._score_candidates(candidates, unary, hidden) is scores
    selector.assert_called_once()


def test_greedy_walk_contract_reference():
    """Pure-Python mirror of ``_selector_walk_kernel_ascend``'s greedy walk
    (the kernel cannot launch on CPU): argmax per step, lowest index wins
    ties, chained via the previous step's winner, and padding rows
    (req_state < 0) never write."""
    num_steps, top_k = 2, 3
    candidates = torch.tensor(
        [[100, 200, 300], [400, 500, 600], [7, 8, 9], [70, 80, 90]],
        dtype=torch.int64,
    )
    scores = torch.full((4, top_k, top_k), 7.0)
    scores[0, 0] = torch.tensor([0.5, 0.9, 0.9])  # tie between 1 and 2
    scores[1, 1] = torch.tensor([-1.0, -2.0, 3.0])  # continues from candidate 1
    req_state = [0, 0, -1, -1]  # row 1 is padding

    tokens = [-123] * 4
    realized = [[-777.0] * top_k for _ in range(4)]
    for row in range(2):
        if req_state[row * num_steps] < 0:
            continue
        previous = 0
        for step in range(num_steps):
            flat = row * num_steps + step
            row_scores = scores[flat, previous].tolist()
            index = row_scores.index(max(row_scores))
            tokens[flat] = candidates[flat, index].item()
            realized[flat] = row_scores
            previous = index

    assert tokens == [200, 600, -123, -123]
    # The fixture stores fp32 scores, so compare with tolerance.
    assert realized[0] == pytest.approx([0.5, 0.9, 0.9])
    assert realized[1] == pytest.approx([-1.0, -2.0, 3.0])
    assert realized[2] == [-777.0] * top_k and realized[3] == [-777.0] * top_k
