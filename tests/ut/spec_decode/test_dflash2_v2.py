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

from contextlib import nullcontext
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


@pytest.mark.parametrize("fail", [False, True])
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
