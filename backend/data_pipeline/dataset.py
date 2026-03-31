"""PyTorch Dataset for Qwen2.5-VL fine-tuning on LEGO image-JSON pairs."""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from backend.config import DATA_DIR, MAX_SEQ_LENGTH
from backend.models.tokenizer import SYSTEM_PROMPT, USER_PROMPT

IMAGE_SIZE = 448  # Qwen2.5-VL dynamic resolution, but we resize for augmentation

# ── Augmentation transforms ───────────────────────────────────────────

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.3, hue=0.05),
])

VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
])


class LegoDataset(Dataset):
    """Dataset that pairs LEGO set images with their JSON descriptions."""

    def __init__(
        self,
        set_nums: list[str],
        data_dir: Path = DATA_DIR,
        processor=None,
        max_length: int = MAX_SEQ_LENGTH,
        split: str = "train",
    ):
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.labels_dir = self.data_dir / "labels"
        self.processor = processor
        self.max_length = max_length
        self.split = split
        self.augment = split == "train"

        # Filter to sets that have both image and label
        self.samples = []
        for set_num in set_nums:
            label_path = self.labels_dir / f"{set_num}.json"
            if not label_path.exists():
                continue
            image_path = self._find_image(set_num)
            if image_path:
                self.samples.append((image_path, label_path))

    def _find_image(self, set_num: str) -> Path | None:
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            path = self.images_dir / f"{set_num}{ext}"
            if path.exists():
                return path
        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        image_path, label_path = self.samples[idx]

        # Load image
        image = Image.open(image_path).convert("RGB")
        if self.augment:
            image = TRAIN_TRANSFORMS(image)
        else:
            image = VAL_TRANSFORMS(image)

        # Load JSON label
        with open(label_path, "r") as f:
            label = json.load(f)
        label_text = json.dumps(label, indent=2)

        if self.processor is not None:
            # Build chat messages for Qwen2.5-VL
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": USER_PROMPT},
                    ],
                },
                {
                    "role": "assistant",
                    "content": label_text,
                },
            ]

            # Apply chat template to get the full text
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )

            # Process with the Qwen processor
            inputs = self.processor(
                text=[text],
                images=[image],
                return_tensors="pt",
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
            )
            # Squeeze batch dim
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}

            # Create labels: copy input_ids, mask everything before the assistant response with -100
            input_ids = inputs["input_ids"]
            labels = input_ids.clone()

            # Find where the assistant response starts by looking for the assistant header token
            # Mask the prompt portion so loss is only computed on the JSON output
            text_prompt_only = self.processor.apply_chat_template(
                messages[:2], tokenize=False, add_generation_prompt=True
            )
            prompt_tokens = self.processor.tokenizer(
                text_prompt_only, return_tensors="pt", truncation=True,
                max_length=self.max_length,
            )
            prompt_len = prompt_tokens["input_ids"].shape[1]

            labels[:prompt_len] = -100
            # Also mask padding tokens
            pad_token_id = self.processor.tokenizer.pad_token_id
            if pad_token_id is not None:
                labels[labels == pad_token_id] = -100

            inputs["labels"] = labels

            return inputs
        else:
            return {"image": image, "label": label, "label_text": label_text}


def load_splits(data_dir: Path = DATA_DIR) -> dict[str, list[str]]:
    """Load train/val split from splits.json."""
    splits_path = data_dir / "splits.json"
    with open(splits_path, "r") as f:
        return json.load(f)


def create_dataloaders(
    processor,
    data_dir: Path = DATA_DIR,
    batch_size: int = 2,
    num_workers: int = 4,
    max_length: int = MAX_SEQ_LENGTH,
) -> tuple:
    """Create train and val DataLoaders."""
    splits = load_splits(data_dir)

    train_ds = LegoDataset(
        set_nums=splits["train"],
        data_dir=data_dir,
        processor=processor,
        max_length=max_length,
        split="train",
    )
    val_ds = LegoDataset(
        set_nums=splits["val"],
        data_dir=data_dir,
        processor=processor,
        max_length=max_length,
        split="val",
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
