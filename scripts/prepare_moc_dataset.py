#!/usr/bin/env python3
"""Augment the LEGO training dataset with style-transferred images.

Creates additional training samples by applying transformations that make
LEGO product photos look more like casual real-world photos:
- Random background changes
- Perspective distortion
- Color/lighting variation
- Crop/zoom to simulate partial views

This teaches the model to handle non-perfect images.

Usage:
    python scripts/prepare_moc_dataset.py --augmentations 3
"""

import argparse
import json
import random
import sys
from pathlib import Path

from PIL import Image, ImageFilter, ImageEnhance, ImageOps
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import DATA_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="Augment LEGO dataset with realistic transforms")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--augmentations", type=int, default=3,
                        help="Number of augmented versions per original image")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def augment_image(img: Image.Image, aug_idx: int) -> Image.Image:
    """Apply realistic augmentations to simulate real-world photos."""
    img = img.copy()
    rng = random.Random(aug_idx)

    transforms = [
        # Simulate different lighting
        lambda im: ImageEnhance.Brightness(im).enhance(rng.uniform(0.6, 1.4)),
        # Simulate different camera white balance
        lambda im: ImageEnhance.Color(im).enhance(rng.uniform(0.5, 1.5)),
        # Simulate phone camera contrast
        lambda im: ImageEnhance.Contrast(im).enhance(rng.uniform(0.7, 1.3)),
        # Slight blur (out of focus)
        lambda im: im.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.5, 1.5))),
        # Random crop (simulate partial view)
        lambda im: _random_crop(im, rng),
        # Add slight rotation
        lambda im: im.rotate(rng.uniform(-15, 15), fillcolor=(240, 240, 240), expand=True),
        # Simulate noisy background
        lambda im: _add_colored_border(im, rng),
        # Sharpen (simulate phone camera sharpening)
        lambda im: ImageEnhance.Sharpness(im).enhance(rng.uniform(0.5, 2.0)),
    ]

    # Apply 2-4 random transforms
    chosen = rng.sample(transforms, k=rng.randint(2, 4))
    for t in chosen:
        try:
            img = t(img)
        except Exception:
            pass

    return img


def _random_crop(img: Image.Image, rng: random.Random) -> Image.Image:
    """Crop to 60-90% of the image from a random position."""
    w, h = img.size
    crop_ratio = rng.uniform(0.6, 0.9)
    new_w = int(w * crop_ratio)
    new_h = int(h * crop_ratio)
    left = rng.randint(0, w - new_w)
    top = rng.randint(0, h - new_h)
    return img.crop((left, top, left + new_w, top + new_h))


def _add_colored_border(img: Image.Image, rng: random.Random) -> Image.Image:
    """Add a colored border to simulate a non-white background."""
    color = (rng.randint(100, 250), rng.randint(100, 250), rng.randint(100, 250))
    border = rng.randint(10, 50)
    return ImageOps.expand(img, border=border, fill=color)


def main():
    args = parse_args()
    random.seed(args.seed)

    data_dir = Path(args.data_dir)
    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"
    splits_path = data_dir / "splits.json"

    if not splits_path.exists():
        print("ERROR: No splits.json found. Run prepare_dataset.py first.")
        sys.exit(1)

    with open(splits_path) as f:
        splits = json.load(f)

    # Only augment training set
    train_sets = splits["train"]
    print(f"Augmenting {len(train_sets)} training samples x{args.augmentations}")

    new_train = []

    for set_num in tqdm(train_sets, desc="Augmenting"):
        # Find original image
        img_path = None
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            p = images_dir / f"{set_num}{ext}"
            if p.exists():
                img_path = p
                break

        label_path = labels_dir / f"{set_num}.json"
        if not img_path or not label_path.exists():
            continue

        # Load original
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        with open(label_path) as f:
            label = json.load(f)

        # Create augmented versions
        for aug_idx in range(args.augmentations):
            aug_id = f"{set_num}_aug{aug_idx}"

            # Save augmented image
            aug_img = augment_image(img, aug_idx=hash(f"{set_num}_{aug_idx}"))
            aug_img_path = images_dir / f"{aug_id}.jpg"
            aug_img.save(aug_img_path, "JPEG", quality=85)

            # Copy label (same LEGO build, different image)
            aug_label_path = labels_dir / f"{aug_id}.json"
            aug_label = label.copy()
            aug_label["set_id"] = aug_id
            with open(aug_label_path, "w") as f:
                json.dump(aug_label, f, indent=2)

            new_train.append(aug_id)

    # Update splits
    splits["train"] = train_sets + new_train
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"\n{'=' * 60}")
    print("Augmentation Complete")
    print(f"{'=' * 60}")
    print(f"  Original training:    {len(train_sets)} samples")
    print(f"  New augmented:        {len(new_train)} samples")
    print(f"  Total training now:   {len(splits['train'])} samples")
    print(f"  Val (unchanged):      {len(splits['val'])} samples")


if __name__ == "__main__":
    main()
