"""Tests for Stage 1 inference.

The real Stage1Pipeline is omitted from coverage because CI does not load the
actual GPU model, but we still exercise its prod-mode wiring with stubbed
dependencies so `LEGOGEN_DEV=0` behavior is pinned."""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.inference.brick_pipeline import _MockStage1
from backend.inference.stage1_pipeline import _strip_thinking_blocks


def test_mock_stage1_describe_returns_string():
    mock = _MockStage1()
    caption = mock.describe(object())
    assert isinstance(caption, str)
    assert len(caption) > 0


def test_mock_stage1_describe_is_stable():
    """Callers rely on the caption being deterministic for dev-mode testing."""
    assert _MockStage1().describe(None) == _MockStage1().describe(None)


def test_strip_thinking_blocks_removes_hidden_reasoning():
    assert _strip_thinking_blocks("<think>plan</think> a red house") == "a red house"


def test_stage1_describe_strips_thinking_and_forwards_should_cancel(monkeypatch):
    import backend.inference.stage1_pipeline as sp

    class FakeInferenceMode:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    fake_torch = types.ModuleType("torch")
    fake_torch.inference_mode = lambda: FakeInferenceMode()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class FakeInputIds:
        shape = (1, 3)

    class FakeInputs(dict):
        def to(self, device):
            self["moved_to"] = device
            return self

    fake_inputs = FakeInputs({"input_ids": FakeInputIds()})
    seen = {}

    class FakeTokenizer:
        def decode(self, generated_ids, skip_special_tokens=True):
            seen["generated_ids"] = generated_ids
            seen["skip_special_tokens"] = skip_special_tokens
            return "<think>outline</think> a red house"

    class FakeProcessor:
        def __init__(self):
            self.tokenizer = FakeTokenizer()

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
            seen["messages"] = messages
            seen["enable_thinking"] = enable_thinking
            return "chat prompt"

        def __call__(self, *, text, images, return_tensors):
            seen["processor_text"] = text
            seen["processor_images"] = images
            seen["processor_return_tensors"] = return_tensors
            return fake_inputs

    class FakeModel:
        device = "cuda:0"

        def generate(self, **kwargs):
            seen["generate_kwargs"] = kwargs
            return [[10, 11, 12, 13, 14]]

    pipe = sp.Stage1Pipeline.__new__(sp.Stage1Pipeline)
    pipe.processor = FakeProcessor()
    pipe.model = FakeModel()

    stopping_criteria = object()
    should_cancel = lambda: False

    def fake_build_stopping_criteria(callback):
        seen["stopping_callback"] = callback
        return stopping_criteria

    monkeypatch.setattr(sp, "build_stopping_criteria", fake_build_stopping_criteria)

    caption = pipe.describe(object(), should_cancel=should_cancel)

    assert caption == "a red house"
    assert seen["enable_thinking"] is False
    assert seen["processor_return_tensors"] == "pt"
    assert fake_inputs["moved_to"] == "cuda:0"
    assert seen["generate_kwargs"]["max_new_tokens"] == sp.STAGE1_MAX_NEW_TOKENS
    assert seen["generate_kwargs"]["temperature"] == sp.STAGE1_TEMPERATURE
    assert seen["generate_kwargs"]["top_p"] == sp.STAGE1_TOP_P
    assert seen["generate_kwargs"]["do_sample"] is True
    assert seen["generate_kwargs"]["stopping_criteria"] is stopping_criteria
    assert seen["stopping_callback"] is should_cancel
    assert seen["generated_ids"] == [13, 14]


def test_stage1_init_wraps_base_model_with_adapter_when_present(monkeypatch, tmp_path):
    import backend.inference.stage1_pipeline as sp

    adapter_dir = tmp_path / "stage1-adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}")

    monkeypatch.setattr(sp, "STAGE1_CHECKPOINT_DIR", adapter_dir)
    monkeypatch.setattr(sp, "USE_BF16", True)

    fake_torch = types.ModuleType("torch")
    fake_torch.bfloat16 = "bf16"
    fake_torch.float16 = "fp16"
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    seen = {}
    fake_processor = object()
    fake_base_model = MagicMock()
    fake_peft_model = MagicMock()

    class FakeBitsAndBytesConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeAutoProcessor:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            seen["processor"] = (model_name, kwargs)
            return fake_processor

    class FakeQwen3_5ForConditionalGeneration:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            seen["model"] = (model_name, kwargs)
            return fake_base_model

    class FakePeftModel:
        @staticmethod
        def from_pretrained(base, path):
            seen["adapter"] = (base, path)
            return fake_peft_model

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoProcessor = FakeAutoProcessor
    fake_transformers.BitsAndBytesConfig = FakeBitsAndBytesConfig
    fake_transformers.Qwen3_5ForConditionalGeneration = FakeQwen3_5ForConditionalGeneration
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    fake_peft = types.ModuleType("peft")
    fake_peft.PeftModel = FakePeftModel
    monkeypatch.setitem(sys.modules, "peft", fake_peft)

    pipe = sp.Stage1Pipeline()

    assert pipe.processor is fake_processor
    assert pipe.model is fake_peft_model
    fake_peft_model.eval.assert_called_once()
    assert seen["processor"][0] == sp.STAGE1_MODEL_NAME
    assert seen["processor"][1]["min_pixels"] == 128 * 28 * 28
    assert seen["processor"][1]["max_pixels"] == 256 * 28 * 28
    assert seen["model"][1]["device_map"] == "auto"
    assert seen["model"][1]["torch_dtype"] == "bf16"
    assert seen["model"][1]["quantization_config"].kwargs["load_in_4bit"] is True
    assert seen["adapter"] == (fake_base_model, str(adapter_dir))
