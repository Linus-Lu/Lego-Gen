"""Tests for Qwen3.5-9B planner migration.

Run:
    pytest tests/test_planner_migration.py -v
    pytest tests/test_planner_migration.py -v -m gpu       # GPU-only tests
    pytest tests/test_planner_migration.py -v -m "not gpu"  # CPU-only tests
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── CPU Tests (no model download needed) ──────────────────────────


def test_planner_config_values():
    """Verify config constants are updated for Qwen3.5-9B."""
    from backend.config import (
        PLANNER_MODEL_NAME,
        PLANNER_LORA_R,
        PLANNER_LORA_ALPHA,
        PLANNER_LORA_DROPOUT,
        PLANNER_LORA_TARGET_MODULES,
        PLANNER_BATCH_SIZE,
        PLANNER_GRADIENT_ACCUMULATION,
        PLANNER_LEARNING_RATE,
        PLANNER_NUM_EPOCHS,
        PLANNER_WARMUP_STEPS,
    )
    assert PLANNER_MODEL_NAME == "Qwen/Qwen3.5-9B"
    assert PLANNER_LORA_R == 64
    assert PLANNER_LORA_ALPHA == 128
    assert PLANNER_LORA_DROPOUT == 0.05
    assert PLANNER_LORA_TARGET_MODULES == "all-linear"
    assert PLANNER_BATCH_SIZE == 2
    assert PLANNER_GRADIENT_ACCUMULATION == 8
    assert PLANNER_LEARNING_RATE == 3e-5
    assert PLANNER_NUM_EPOCHS == 5
    assert PLANNER_WARMUP_STEPS == 300


def test_system_prompt_disables_thinking():
    """Verify PLANNER_SYSTEM_PROMPT includes /no_think directive."""
    from backend.models.tokenizer import PLANNER_SYSTEM_PROMPT

    assert "/no_think" in PLANNER_SYSTEM_PROMPT


def test_strip_thinking_blocks():
    """Verify thinking block removal works correctly."""
    from backend.models.tokenizer import strip_thinking_blocks

    # Normal case
    text = '<think>reasoning here</think>{"key": "value"}'
    assert strip_thinking_blocks(text) == '{"key": "value"}'

    # No thinking block
    text = '{"key": "value"}'
    assert strip_thinking_blocks(text) == '{"key": "value"}'

    # Multi-line thinking
    text = '<think>\nlong\nreasoning\n</think>\n{"key": "value"}'
    result = strip_thinking_blocks(text)
    assert "<think>" not in result
    assert '{"key": "value"}' in result

    # Empty thinking block
    text = '<think></think>{"key": "value"}'
    assert strip_thinking_blocks(text) == '{"key": "value"}'


# ── GPU Tests (require CUDA + model download) ────────────────────


@pytest.mark.gpu
def test_model_loads_with_qlora():
    """Verify Qwen3.5-9B loads with 4-bit quantization and LoRA."""
    from backend.models.planner_lm import LegoPlannerLM

    planner = LegoPlannerLM()
    model = planner.get_model()
    tokenizer = planner.get_tokenizer()

    assert model is not None
    assert tokenizer is not None
    assert tokenizer.pad_token is not None


@pytest.mark.gpu
def test_lora_applied_to_all_linear():
    """Verify LoRA adapters exist on linear layers including DeltaNet layers."""
    from backend.models.planner_lm import LegoPlannerLM

    planner = LegoPlannerLM()
    model = planner.get_model()

    # Count trainable params -- should be > 0
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    assert trainable > 0
    assert trainable / total < 0.10  # LoRA should be < 10% of total params

    # Check that lora_A/lora_B appear in named parameters
    lora_params = [n for n, _ in model.named_parameters() if "lora_" in n]
    assert len(lora_params) > 0

    # With all-linear at rank 64, we expect broad coverage across layers
    assert len(lora_params) > 50


@pytest.mark.gpu
def test_tokenizer_chat_template():
    """Verify chat template produces valid token sequences with thinking disabled."""
    from backend.models.planner_lm import LegoPlannerLM
    from backend.models.tokenizer import build_planner_chat_messages

    planner = LegoPlannerLM()
    tokenizer = planner.get_tokenizer()

    messages = build_planner_chat_messages("Build me a red house")
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )

    assert isinstance(text, str)
    assert len(text) > 0
    assert "<think>" not in text
    assert "red house" in text


@pytest.mark.gpu
def test_forward_pass_produces_gradients():
    """Verify a forward pass produces valid gradients on LoRA parameters."""
    import torch
    from backend.models.planner_lm import LegoPlannerLM

    planner = LegoPlannerLM()
    model = planner.get_model()
    tokenizer = planner.get_tokenizer()

    # Create a dummy input
    inputs = tokenizer("Build me a LEGO house", return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    inputs["labels"] = inputs["input_ids"].clone()

    # Forward pass
    model.train()
    outputs = model(**inputs)
    loss = outputs.loss

    assert loss is not None
    assert loss.item() > 0

    # Backward pass
    loss.backward()

    # Check at least some LoRA params have gradients
    has_grad = any(
        p.grad is not None
        for n, p in model.named_parameters()
        if "lora_" in n
    )
    assert has_grad


@pytest.mark.gpu
def test_no_thinking_tokens_in_output():
    """Verify that generation does not produce <think> tokens."""
    import torch
    from backend.models.planner_lm import LegoPlannerLM
    from backend.models.tokenizer import build_planner_chat_messages

    planner = LegoPlannerLM()
    model = planner.get_model()
    tokenizer = planner.get_tokenizer()

    messages = build_planner_chat_messages("Build me a small red car")
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    model.eval()
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)

    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False
    )
    assert "<think>" not in generated


@pytest.mark.gpu
def test_single_training_step():
    """Smoke test: run one training step without errors."""
    import torch
    from transformers import TrainingArguments, Trainer
    from backend.models.planner_lm import LegoPlannerLM

    planner = LegoPlannerLM()
    model = planner.get_model()
    tokenizer = planner.get_tokenizer()

    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, idx):
            text = '{"object": "house", "category": "City"}'
            enc = tokenizer(
                text, return_tensors="pt", padding="max_length",
                max_length=64, truncation=True,
            )
            result = {k: v.squeeze(0) for k, v in enc.items()}
            result["labels"] = enc["input_ids"].squeeze(0)
            return result

    args = TrainingArguments(
        output_dir="/tmp/test_planner",
        per_device_train_batch_size=2,
        max_steps=1,
        no_cuda=False,
        report_to="none",
        gradient_checkpointing=False,
        remove_unused_columns=False,
    )

    trainer = Trainer(model=model, args=args, train_dataset=DummyDataset())
    result = trainer.train()

    assert result.training_loss > 0
