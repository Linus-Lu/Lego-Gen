"""Stage 1 dataset: image → natural-language description for LEGO building.

Each sample pairs a photo of an object with a short text description.
The model is fine-tuned to produce LEGO-relevant geometry descriptions
from images (Stage 1 of the two-stage pipeline).
"""

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.config import (
    STAGE1_MAX_SEQ_LENGTH,
    STAGE1_SYSTEM_PROMPT,
)
from backend.data_pipeline.dataset import TRAIN_TRANSFORMS, VAL_TRANSFORMS

# Fixed user turn message for Stage 1
STAGE1_USER_PROMPT = "Describe this object for LEGO building."


class Stage1Dataset(Dataset):
    """Dataset that pairs object images with short LEGO-relevant descriptions.

    Reads a manifest JSON file of the form::

        [{"image_path": "...", "description": "...", "category": "...", "source": "..."}, ...]

    Applies a 90/10 train/val split by default.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        processor,
        max_length: int = STAGE1_MAX_SEQ_LENGTH,
        split: str = "train",
    ):
        self.processor = processor
        self.max_length = max_length
        self.split = split
        self.augment = split == "train"

        manifest_path = Path(manifest_path)
        with open(manifest_path, "r") as f:
            all_records = json.load(f)

        # Deterministic 90/10 train/val split
        rng = random.Random(42)
        shuffled = list(all_records)
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * 0.1))
        if split == "train":
            self.records = shuffled[n_val:]
        else:
            self.records = shuffled[:n_val]

        print(f"  [Stage1Dataset/{split}] {len(self.records)} samples "
              f"(from manifest of {len(all_records)})")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]

        # ── Load and optionally augment image ─────────────────────────
        image = Image.open(record["image_path"]).convert("RGB")
        if self.augment:
            image = TRAIN_TRANSFORMS(image)
        else:
            image = VAL_TRANSFORMS(image)

        description = record["description"]

        # ── Build chat messages ────────────────────────────────────────
        messages = [
            {"role": "system", "content": STAGE1_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": STAGE1_USER_PROMPT},
                ],
            },
            {"role": "assistant", "content": description},
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )

        # Squeeze batch dim from text tensors; keep vision tensors intact
        no_squeeze = {"pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"}
        inputs = {
            k: v.squeeze(0) if k not in no_squeeze else v
            for k, v in inputs.items()
        }

        # Mask prompt tokens with -100 so loss is only on the description
        inputs["labels"] = self._mask_prompt(inputs)
        return inputs

    def _mask_prompt(self, inputs: dict) -> torch.Tensor:
        """Create labels tensor with all tokens before the assistant response masked.

        Finds the last occurrence of ``<|im_start|>assistant\\n`` in the
        tokenized input_ids and masks every position before (and including)
        that marker with -100.  This matches the pattern used in
        ``dataset_unified.py``.
        """
        input_ids = inputs["input_ids"]
        labels = input_ids.clone()

        # Locate assistant turn start marker
        assistant_marker = self.processor.tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False,
        )
        marker_len = len(assistant_marker)
        marker_t = torch.tensor(assistant_marker, dtype=input_ids.dtype)

        # Search for the last occurrence of the marker in input_ids
        prompt_len = 0
        for i in range(len(input_ids) - marker_len, -1, -1):
            if torch.equal(input_ids[i: i + marker_len], marker_t):
                prompt_len = i + marker_len
                break

        labels[:prompt_len] = -100

        # Mask padding tokens
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100

        return labels


class Stage1Collator:
    """Data collator for Stage1Dataset batches.

    - Stacks ``input_ids``, ``attention_mask``, and ``labels`` tensors.
    - Concatenates ``pixel_values`` and ``image_grid_thw`` from vision samples.
    """

    def __call__(self, features: list[dict]) -> dict:
        batch = {}

        # Standard sequence tensors (mm_token_type_ids required by Qwen3.5 M-RoPE)
        for key in ("input_ids", "attention_mask", "labels", "mm_token_type_ids"):
            if key in features[0]:
                batch[key] = torch.stack([f[key] for f in features])

        # Vision tensors — all Stage 1 samples have images
        has_pixels = [f for f in features if "pixel_values" in f]
        if has_pixels:
            batch["pixel_values"] = torch.cat(
                [f["pixel_values"] for f in has_pixels], dim=0
            )
        if any("image_grid_thw" in f for f in features):
            grids = [f["image_grid_thw"] for f in features if "image_grid_thw" in f]
            batch["image_grid_thw"] = torch.cat(grids, dim=0)

        return batch
