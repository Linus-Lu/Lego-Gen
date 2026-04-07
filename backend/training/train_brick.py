"""Fine-tune Qwen3.5-4B with LoRA for text → colored brick sequence generation."""

import inspect
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

# Check if SFTConfig exists (TRL >= 0.12)
try:
    from trl import SFTConfig
except ImportError:
    SFTConfig = None

from backend.config import (
    BRICK_MODEL_NAME, BRICK_CHECKPOINT_DIR, BRICK_LEARNING_RATE,
    BRICK_BATCH_SIZE, BRICK_GRADIENT_ACCUMULATION, BRICK_MAX_SEQ_LENGTH,
    BRICK_NUM_EPOCHS, BRICK_LORA_R, BRICK_LORA_ALPHA, BRICK_LORA_DROPOUT,
    BRICK_TRAINING_DATA,
)


def _inspect_params(cls):
    """Return set of parameter names accepted by cls.__init__."""
    try:
        return set(inspect.signature(cls.__init__).parameters.keys())
    except (ValueError, TypeError):
        return set()


def main() -> None:
    train_path = BRICK_TRAINING_DATA / "train.jsonl"
    test_path = BRICK_TRAINING_DATA / "test.jsonl"

    if not train_path.exists():
        print(f"ERROR: {train_path} not found. Run prepare_brick_dataset.py first.")
        sys.exit(1)

    # Print TRL/transformers versions for debugging
    import trl, transformers
    print(f"TRL version: {trl.__version__}", flush=True)
    print(f"Transformers version: {transformers.__version__}", flush=True)

    print(f"Loading model: {BRICK_MODEL_NAME}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BRICK_MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BRICK_MODEL_NAME, torch_dtype="auto", trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=BRICK_LORA_R, lora_alpha=BRICK_LORA_ALPHA,
        lora_dropout=BRICK_LORA_DROPOUT,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    print("Loading datasets...", flush=True)
    ds = load_dataset("json", data_files={
        "train": str(train_path),
        "test": str(test_path),
    })

    output_dir = str(BRICK_CHECKPOINT_DIR)

    # ── Build training args (version-adaptive) ────────────────────────
    base_kwargs = dict(
        output_dir=output_dir,
        num_train_epochs=BRICK_NUM_EPOCHS,
        per_device_train_batch_size=BRICK_BATCH_SIZE,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=BRICK_GRADIENT_ACCUMULATION,
        learning_rate=BRICK_LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=500,
        report_to="none",
        dataloader_pin_memory=False,
    )

    # Decide which config class to use
    ConfigClass = SFTConfig if SFTConfig is not None else TrainingArguments
    config_params = _inspect_params(ConfigClass)

    config_kwargs = dict(base_kwargs)
    if "max_seq_length" in config_params:
        config_kwargs["max_seq_length"] = BRICK_MAX_SEQ_LENGTH
    if "dataset_text_field" in config_params:
        config_kwargs["dataset_text_field"] = None

    print(f"Using config class: {ConfigClass.__name__}", flush=True)
    training_args = ConfigClass(**config_kwargs)

    # ── Build trainer (version-adaptive) ──────────────────────────────
    trainer_params = _inspect_params(SFTTrainer)
    print(f"SFTTrainer accepts: processing_class={'processing_class' in trainer_params}, "
          f"tokenizer={'tokenizer' in trainer_params}, "
          f"max_seq_length={'max_seq_length' in trainer_params}", flush=True)

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
    )

    # Pass tokenizer with the right param name
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer

    # Pass max_seq_length to trainer if config didn't accept it
    if "max_seq_length" in trainer_params and "max_seq_length" not in config_kwargs:
        trainer_kwargs["max_seq_length"] = BRICK_MAX_SEQ_LENGTH

    trainer = SFTTrainer(**trainer_kwargs)

    print("Starting training...", flush=True)
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
