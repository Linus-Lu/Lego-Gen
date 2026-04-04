"""Unified dataset for mixed vision (image→JSON) and planner (text→JSON) training.

Interleaves image-JSON pairs with text-JSON pairs so a single LoRA adapter
learns both tasks on one Qwen3.5-9B model.
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from backend.config import (
    DATA_DIR,
    UNIFIED_MAX_SEQ_LENGTH,
    PLANNER_PROMPTS_DIR,
    ST2B_CONVERTED_DIR,
    ST2B_PROMPTS_DIR,
    VISION_UPSAMPLE,
)
from backend.models.tokenizer import (
    SYSTEM_PROMPT,
    USER_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    sample_prompt_template,
)
from backend.data_pipeline.dataset import TRAIN_TRANSFORMS, VAL_TRANSFORMS


class UnifiedLegoDataset(Dataset):
    """Mixed dataset: image-JSON (vision) + text-JSON (planner) samples."""

    def __init__(
        self,
        vision_set_nums: list[str] | None = None,
        rebrickable_ids: list[str] | None = None,
        st2b_ids: list[str] | None = None,
        data_dir: Path = DATA_DIR,
        processor=None,
        max_length: int = UNIFIED_MAX_SEQ_LENGTH,
        split: str = "train",
        epoch: int = 0,
        rebrickable_upsample: int = 3,
        vision_upsample: int = VISION_UPSAMPLE,
    ):
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.max_length = max_length
        self.split = split
        self.epoch = epoch
        self.augment = split == "train"

        self.samples: list[tuple[str, dict]] = []  # (type, paths_dict)

        # ── Vision samples (image + JSON) ─────────────────────────────
        if vision_set_nums:
            images_dir = self.data_dir / "images"
            labels_dir = self.data_dir / "labels"
            vision_samples = []
            for set_num in vision_set_nums:
                label_path = labels_dir / f"{set_num}.json"
                if not label_path.exists():
                    continue
                image_path = self._find_image(images_dir, set_num)
                if image_path:
                    vision_samples.append(("vision", {
                        "image_path": image_path,
                        "label_path": label_path,
                    }))
            # Upsample vision to balance with planner data
            self.samples.extend(vision_samples * max(1, vision_upsample))

        # ── Rebrickable text samples ──────────────────────────────────
        if rebrickable_ids:
            labels_dir = self.data_dir / "labels"
            rb_samples = []
            for sid in rebrickable_ids:
                label_path = labels_dir / f"{sid}.json"
                if not label_path.exists():
                    continue
                rb_samples.append(("text", {
                    "label_path": label_path,
                    "prompts_path": PLANNER_PROMPTS_DIR / f"{sid}.json",
                    "source": "rebrickable",
                }))
            self.samples.extend(rb_samples * max(1, rebrickable_upsample))

        # ── StableText2Brick text samples ─────────────────────────────
        if st2b_ids:
            for sid in st2b_ids:
                label_path = ST2B_CONVERTED_DIR / f"{sid}.json"
                if not label_path.exists():
                    continue
                self.samples.append(("text", {
                    "label_path": label_path,
                    "prompts_path": ST2B_PROMPTS_DIR / f"{sid}.json",
                    "source": "st2b",
                }))

    @staticmethod
    def _find_image(images_dir: Path, set_num: str) -> Path | None:
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            path = images_dir / f"{set_num}{ext}"
            if path.exists():
                try:
                    img = Image.open(path)
                    img.verify()
                    return path
                except Exception:
                    continue
        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample_type, paths = self.samples[idx]

        if sample_type == "vision":
            return self._get_vision_sample(paths)
        else:
            return self._get_text_sample(paths, idx)

    def _get_vision_sample(self, paths: dict) -> dict:
        """Process an image-JSON sample."""
        image = Image.open(paths["image_path"]).convert("RGB")
        if self.augment:
            image = TRAIN_TRANSFORMS(image)
        else:
            image = VAL_TRANSFORMS(image)

        with open(paths["label_path"], "r") as f:
            label = json.load(f)
        label_text = json.dumps(label)

        # Build vision chat messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": USER_PROMPT},
                ],
            },
            {"role": "assistant", "content": label_text},
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
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
        # Squeeze batch dim from text tensors only; pixel_values and
        # image_grid_thw must keep their leading dimensions intact.
        no_squeeze = {"pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"}
        inputs = {k: v.squeeze(0) if k not in no_squeeze else v for k, v in inputs.items()}

        # Mask prompt tokens with -100
        inputs["labels"] = self._mask_prompt(inputs, messages[:2])
        return inputs

    def _get_text_sample(self, paths: dict, idx: int) -> dict:
        """Process a text-JSON sample."""
        with open(paths["label_path"], "r") as f:
            label = json.load(f)
        label_text = json.dumps(label, indent=2)

        # Pick a prompt with rotation
        prompt = self._pick_prompt(label, paths.get("prompts_path"), idx)

        # Build planner chat messages
        messages = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": label_text},
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
            enable_thinking=False,
        )

        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # Mask prompt tokens with -100
        inputs["labels"] = self._mask_prompt(inputs, messages[:2])
        return inputs

    def _mask_prompt(self, inputs: dict, prompt_messages: list) -> torch.Tensor:
        """Create labels tensor with prompt tokens masked as -100.

        Finds the assistant turn marker in the tokenized input_ids rather than
        re-tokenizing the prompt separately, which avoids length mismatches
        caused by image tokens inserted by the processor.
        """
        input_ids = inputs["input_ids"]
        labels = input_ids.clone()

        # Find the assistant turn start marker in the token IDs.
        # Qwen chat template uses "<|im_start|>assistant\n" before the response.
        assistant_marker = self.processor.tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False,
        )
        marker_len = len(assistant_marker)
        marker_t = torch.tensor(assistant_marker, dtype=input_ids.dtype)

        # Search for the last occurrence of the marker in input_ids
        prompt_len = 0
        for i in range(len(input_ids) - marker_len, -1, -1):
            if torch.equal(input_ids[i : i + marker_len], marker_t):
                prompt_len = i + marker_len
                break

        labels[:prompt_len] = -100

        # Mask padding tokens
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100

        return labels

    def _pick_prompt(self, label: dict, prompts_path: Path | None, idx: int) -> str:
        """Pick a prompt template with epoch-based rotation."""
        if prompts_path and prompts_path.exists():
            try:
                with open(prompts_path, "r") as f:
                    prompts_data = json.load(f)
                prompts = prompts_data if isinstance(prompts_data, list) else prompts_data.get("prompts", [])
                if prompts:
                    return prompts[(self.epoch + idx) % len(prompts)]
            except (json.JSONDecodeError, KeyError):
                pass
        return sample_prompt_template(label, epoch=self.epoch)


class UnifiedCollator:
    """Data collator that handles mixed batches with optional pixel_values.

    With batch_size=1 (default), this is straightforward — each sample is
    either vision or text. For batch_size > 1, concatenates pixel_values
    from vision samples and passes None when all samples are text-only.
    """

    def __call__(self, features: list[dict]) -> dict:
        batch = {}

        # Standard sequence-length tensor fields — stack across batch
        for key in ("input_ids", "attention_mask", "labels", "mm_token_type_ids"):
            if key in features[0]:
                batch[key] = torch.stack([f[key] for f in features])

        # Vision-specific fields — only present in image samples
        has_pixels = [f for f in features if "pixel_values" in f]
        if has_pixels:
            batch["pixel_values"] = torch.cat([f["pixel_values"] for f in has_pixels], dim=0)
        if any("image_grid_thw" in f for f in features):
            grids = [f["image_grid_thw"] for f in features if "image_grid_thw" in f]
            batch["image_grid_thw"] = torch.cat(grids, dim=0)

        return batch


def load_unified_splits(data_dir: Path = DATA_DIR) -> dict:
    """Load all splits: vision + rebrickable + st2b."""
    splits = {}

    # Vision splits
    vision_splits_path = data_dir / "splits.json"
    if vision_splits_path.exists():
        with open(vision_splits_path, "r") as f:
            vs = json.load(f)
        splits["vision_train"] = vs.get("train", [])
        splits["vision_val"] = vs.get("val", [])
    else:
        splits["vision_train"] = []
        splits["vision_val"] = []

    # Rebrickable splits — use dedicated file if available, else filter
    # vision splits to only include IDs that have planner prompts
    rebrickable_splits_path = data_dir / "rebrickable_splits.json"
    if rebrickable_splits_path.exists():
        with open(rebrickable_splits_path, "r") as f:
            rs = json.load(f)
        splits["rebrickable_train"] = rs.get("train", [])
        splits["rebrickable_val"] = rs.get("val", [])
    else:
        # Filter to set IDs that have prompt files (actual planner data)
        prompts_dir = PLANNER_PROMPTS_DIR
        splits["rebrickable_train"] = [
            sid for sid in splits["vision_train"]
            if (prompts_dir / f"{sid}.json").exists()
        ]
        splits["rebrickable_val"] = [
            sid for sid in splits["vision_val"]
            if (prompts_dir / f"{sid}.json").exists()
        ]

    # ST2B splits
    st2b_train_path = data_dir / "st2b_train_split.json"
    st2b_test_path = data_dir / "st2b_test_split.json"

    if st2b_train_path.exists():
        with open(st2b_train_path, "r") as f:
            splits["st2b_train"] = json.load(f).get("ids", [])
    else:
        splits["st2b_train"] = []

    if st2b_test_path.exists():
        with open(st2b_test_path, "r") as f:
            splits["st2b_val"] = json.load(f).get("ids", [])
    else:
        splits["st2b_val"] = []

    return splits
