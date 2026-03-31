"""PyTorch Dataset for BLIP-2 fine-tuning on LEGO image-JSON pairs."""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from backend.config import DATA_DIR, IMAGE_SIZE, MAX_SEQ_LENGTH


# ── Augmentation transforms ───────────────────────────────────────────

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.3, hue=0.05),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
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
            # Find matching image (could be .jpg or .png)
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

        # Process with BLIP-2 processor
        if self.processor is not None:
            from backend.models.tokenizer import get_json_prompt

            prompt = get_json_prompt()
            inputs = self.processor(
                images=image,
                text=prompt,
                return_tensors="pt",
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
            )
            # Squeeze batch dim added by processor
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}

            # Tokenize the label as target
            labels = self.processor.tokenizer(
                label_text,
                return_tensors="pt",
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
            )
            inputs["labels"] = labels["input_ids"].squeeze(0)
            # Mask padding tokens in labels with -100 so they're ignored in loss
            pad_token_id = self.processor.tokenizer.pad_token_id
            inputs["labels"][inputs["labels"] == pad_token_id] = -100

            return inputs
        else:
            # Return raw data (useful for inspection)
            return {"image": image, "label": label, "label_text": label_text}


def load_splits(data_dir: Path = DATA_DIR) -> dict[str, list[str]]:
    """Load train/val split from splits.json."""
    splits_path = data_dir / "splits.json"
    with open(splits_path, "r") as f:
        return json.load(f)


def create_dataloaders(
    processor,
    data_dir: Path = DATA_DIR,
    batch_size: int = 4,
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
