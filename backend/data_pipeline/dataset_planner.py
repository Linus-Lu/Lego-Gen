"""PyTorch Dataset for text-to-JSON fine-tuning combining Rebrickable + StableText2Brick data."""

import json
from pathlib import Path
from random import Random

import torch
from torch.utils.data import Dataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from backend.config import (
    DATA_DIR,
    PLANNER_MAX_SEQ_LENGTH,
    LABELS_DIR,
    PLANNER_PROMPTS_DIR,
    ST2B_CONVERTED_DIR,
    ST2B_PROMPTS_DIR,
)
from backend.models.tokenizer import PLANNER_SYSTEM_PROMPT, sample_prompt_template


class PlannerDataset(Dataset):
    """Dataset that pairs text prompts with JSON descriptions from both sources."""

    def __init__(
        self,
        rebrickable_ids: list[str] | None = None,
        st2b_ids: list[str] | None = None,
        data_dir: Path = DATA_DIR,
        tokenizer=None,
        max_length: int = PLANNER_MAX_SEQ_LENGTH,
        split: str = "train",
        epoch: int = 0,
        rebrickable_upsample: int = 3,
    ):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.epoch = epoch

        # Build sample list: (label_path, prompts_path, source)
        self.samples = []

        # Rebrickable samples (upsampled)
        if rebrickable_ids:
            rb_samples = []
            for sid in rebrickable_ids:
                label_path = LABELS_DIR / f"{sid}.json"
                prompts_path = PLANNER_PROMPTS_DIR / f"{sid}.json"
                if label_path.exists():
                    rb_samples.append((label_path, prompts_path, "rebrickable"))
            # Upsample Rebrickable data
            for _ in range(rebrickable_upsample):
                self.samples.extend(rb_samples)

        # ST2B samples (1x)
        if st2b_ids:
            for sid in st2b_ids:
                label_path = ST2B_CONVERTED_DIR / f"{sid}.json"
                prompts_path = ST2B_PROMPTS_DIR / f"{sid}.json"
                if label_path.exists():
                    self.samples.append((label_path, prompts_path, "st2b"))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        label_path, prompts_path, source = self.samples[idx]

        # Load label JSON
        with open(label_path) as f:
            label = json.load(f)
        label_text = json.dumps(label, indent=2)

        # Pick a prompt
        prompt = self._pick_prompt(label, prompts_path, idx)

        if self.tokenizer is not None:
            return self._tokenize(prompt, label_text, idx)
        else:
            return {"prompt": prompt, "label": label, "label_text": label_text, "source": source}

    def _pick_prompt(self, label: dict, prompts_path: Path, idx: int) -> str:
        """Select a prompt variant, rotating by epoch."""
        # Try loading pre-generated prompts
        if prompts_path.exists():
            with open(prompts_path) as f:
                prompts = json.load(f)
            if prompts:
                prompt_idx = (self.epoch + idx) % len(prompts)
                return prompts[prompt_idx]

        # Fallback: generate on the fly
        return sample_prompt_template(label, epoch=self.epoch)

    def _tokenize(self, prompt: str, label_text: str, idx: int) -> dict:
        """Tokenize prompt + label for causal LM training."""
        # Build chat messages
        messages = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": label_text},
        ]

        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
            enable_thinking=False,
        )

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # Create labels: mask prompt portion with -100
        input_ids = inputs["input_ids"]
        labels = input_ids.clone()

        # Get prompt-only length (system + user, no assistant)
        prompt_messages = messages[:2]
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_tokens = self.tokenizer(
            prompt_text, return_tensors="pt", truncation=True,
            max_length=self.max_length,
        )
        prompt_len = prompt_tokens["input_ids"].shape[1]

        labels[:prompt_len] = -100

        # Mask padding using attention_mask (not pad_token_id, which equals
        # eos_token_id and would also mask the legitimate end-of-sequence token)
        attention_mask = inputs["attention_mask"]
        labels[attention_mask == 0] = -100

        inputs["labels"] = labels
        return inputs


def load_planner_splits(data_dir: Path = DATA_DIR) -> dict:
    """Load train/val splits for both Rebrickable and ST2B data."""
    splits = {"rebrickable_train": [], "rebrickable_val": [],
              "st2b_train": [], "st2b_val": []}

    # Rebrickable splits
    rb_splits_path = data_dir / "splits.json"
    if rb_splits_path.exists():
        with open(rb_splits_path) as f:
            rb = json.load(f)
        splits["rebrickable_train"] = rb.get("train", [])
        splits["rebrickable_val"] = rb.get("val", [])

    # ST2B splits
    st2b_train_path = data_dir / "st2b_train_split.json"
    if st2b_train_path.exists():
        with open(st2b_train_path) as f:
            st = json.load(f)
        splits["st2b_train"] = st.get("ids", [])

    st2b_test_path = data_dir / "st2b_test_split.json"
    if st2b_test_path.exists():
        with open(st2b_test_path) as f:
            st = json.load(f)
        splits["st2b_val"] = st.get("ids", [])

    return splits


def create_planner_dataloaders(
    tokenizer,
    data_dir: Path = DATA_DIR,
    batch_size: int = 2,
    num_workers: int = 4,
    max_length: int = PLANNER_MAX_SEQ_LENGTH,
    rebrickable_upsample: int = 3,
) -> tuple:
    """Create train and val DataLoaders for planner training."""
    splits = load_planner_splits(data_dir)

    train_ds = PlannerDataset(
        rebrickable_ids=splits["rebrickable_train"],
        st2b_ids=splits["st2b_train"],
        data_dir=data_dir,
        tokenizer=tokenizer,
        max_length=max_length,
        split="train",
        rebrickable_upsample=rebrickable_upsample,
    )
    val_ds = PlannerDataset(
        rebrickable_ids=splits["rebrickable_val"],
        st2b_ids=splits["st2b_val"],
        data_dir=data_dir,
        tokenizer=tokenizer,
        max_length=max_length,
        split="val",
        rebrickable_upsample=1,  # no upsampling for val
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
