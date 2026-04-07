"""Fine-tune Qwen3.5-4B with LoRA for text → colored brick sequence generation."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from backend.config import (
    BRICK_MODEL_NAME, BRICK_CHECKPOINT_DIR, BRICK_LEARNING_RATE,
    BRICK_BATCH_SIZE, BRICK_GRADIENT_ACCUMULATION, BRICK_MAX_SEQ_LENGTH,
    BRICK_NUM_EPOCHS, BRICK_LORA_R, BRICK_LORA_ALPHA, BRICK_LORA_DROPOUT,
    BRICK_TRAINING_DATA,
)


def main() -> None:
    train_path = BRICK_TRAINING_DATA / "train.jsonl"
    test_path = BRICK_TRAINING_DATA / "test.jsonl"

    if not train_path.exists():
        print(f"ERROR: {train_path} not found. Run prepare_brick_dataset.py first.")
        sys.exit(1)

    print(f"Loading model: {BRICK_MODEL_NAME}")
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

    # Enable gradient checkpointing to reduce VRAM usage (~30% savings)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    print("Loading datasets...")
    ds = load_dataset("json", data_files={
        "train": str(train_path),
        "test": str(test_path),
    })

    output_dir = str(BRICK_CHECKPOINT_DIR)

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=BRICK_NUM_EPOCHS,
        per_device_train_batch_size=BRICK_BATCH_SIZE,
        per_device_eval_batch_size=1,  # eval with BS=1 to avoid OOM
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
        eval_on_start=False,
        max_seq_length=BRICK_MAX_SEQ_LENGTH,
        dataset_text_field=None,
        report_to="none",
        dataloader_pin_memory=False,  # reduce host memory pressure
    )

    trainer = SFTTrainer(
        model=model, args=training_args,
        train_dataset=ds["train"], eval_dataset=ds["test"],
        processing_class=tokenizer,
    )

    print("Starting training...")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    main()
