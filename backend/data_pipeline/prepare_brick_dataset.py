"""Prepare brick training data from StableText2Brick (HuggingFace dataset).

Downloads the dataset, assigns deterministic colors to monochrome bricks,
and writes JSONL training/test files for fine-tuning.

Usage (do NOT run during tests — downloads the dataset):
    python -m backend.data_pipeline.prepare_brick_dataset
"""

from __future__ import annotations

import hashlib
import json
import re
import random
from pathlib import Path
from typing import NamedTuple

from backend.brick.parser import Brick, serialize_brick

# ---------------------------------------------------------------------------
# Color maps
# ---------------------------------------------------------------------------

COLOR_WORDS: dict[str, str] = {
    "red": "C91A09",
    "blue": "0055BF",
    "green": "237841",
    "yellow": "FEC401",
    "black": "05131D",
    "white": "FFFFFF",
    "brown": "583927",
    "gray": "A0A5A9",
    "grey": "A0A5A9",
    "orange": "FE8A18",
    "pink": "C870A0",
    "purple": "81007B",
    "tan": "E4CD9E",
}

DEFAULT_PALETTE: list[str] = [
    "C91A09", "0055BF", "237841", "FEC401",
    "05131D", "FFFFFF", "583927", "A0A5A9",
]

# Dark colors for ground-level weight boost
_DARK_COLORS: set[str] = {"05131D", "583927", "A0A5A9", "237841", "C91A09"}

# ShapeNet category IDs → human names
SHAPENET_CATEGORIES: dict[str, str] = {
    "02801938": "basket",   "02818832": "bed",      "02828884": "bench",
    "02843684": "birdhouse","02871439": "bookshelf", "02876657": "bottle",
    "02880940": "bowl",     "02924116": "bus",       "02942699": "camera",
    "02958343": "car",      "03001627": "chair",     "03467517": "guitar",
    "03593526": "jar",      "03797390": "mug",       "03928116": "piano",
    "04004475": "pot",      "04256520": "sofa",      "04379243": "table",
    "04460130": "tower",    "04468005": "train",     "04530566": "vessel",
}

CATEGORY_PALETTES: dict[str, list[str]] = {
    "basket":    ["583927", "E4CD9E", "958A73", "C91A09", "FEC401"],
    "bed":       ["FFFFFF", "A0A5A9", "C91A09", "0055BF", "E4CD9E"],
    "bench":     ["583927", "958A73", "A0A5A9", "05131D", "E4CD9E"],
    "birdhouse": ["583927", "E4CD9E", "FFFFFF", "FEC401", "C91A09"],
    "bookshelf": ["583927", "958A73", "E4CD9E", "05131D", "FFFFFF"],
    "bottle":    ["237841", "0055BF", "C91A09", "A0A5A9", "FFFFFF"],
    "bowl":      ["FFFFFF", "A0A5A9", "C91A09", "0055BF", "FEC401"],
    "bus":       ["FEC401", "C91A09", "0055BF", "FFFFFF", "05131D"],
    "camera":    ["05131D", "A0A5A9", "FFFFFF", "C91A09", "0055BF"],
    "car":       ["C91A09", "0055BF", "05131D", "FFFFFF", "A0A5A9"],
    "chair":     ["583927", "958A73", "E4CD9E", "C91A09", "05131D"],
    "guitar":    ["583927", "E4CD9E", "958A73", "C91A09", "05131D"],
    "jar":       ["FFFFFF", "A0A5A9", "C91A09", "237841", "0055BF"],
    "mug":       ["FFFFFF", "C91A09", "0055BF", "FEC401", "A0A5A9"],
    "piano":     ["05131D", "FFFFFF", "A0A5A9", "958A73", "583927"],
    "pot":       ["C91A09", "237841", "958A73", "FEC401", "05131D"],
    "sofa":      ["C91A09", "0055BF", "958A73", "583927", "A0A5A9"],
    "table":     ["583927", "E4CD9E", "958A73", "05131D"],
    "tower":     ["A0A5A9", "958A73", "FFFFFF", "E4CD9E", "05131D"],
    "train":     ["C91A09", "05131D", "0055BF", "A0A5A9", "FEC401"],
    "vessel":    ["A0A5A9", "0055BF", "FFFFFF", "C91A09", "05131D"],
}

# ---------------------------------------------------------------------------
# ST2B brick format: "hxw (x,y,z)"  (no color)
# ---------------------------------------------------------------------------

_ST2B_RE = re.compile(r"^(\d+)x(\d+)\s+\((-?\d+),(-?\d+),(-?\d+)\)$")


def parse_st2b_bricks(raw: str) -> list[tuple[int, int, int, int, int]]:
    """Parse original StableText2Brick format (no color).

    Each line looks like ``"1x4 (15,13,0)"``.

    Returns a list of ``(h, w, x, y, z)`` tuples.
    Empty or whitespace-only lines are silently skipped.
    """
    result: list[tuple[int, int, int, int, int]] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        m = _ST2B_RE.fullmatch(stripped)
        if m is None:
            continue  # skip unparseable lines
        h, w, x, y, z = (int(m.group(i)) for i in range(1, 6))
        result.append((h, w, x, y, z))
    return result


# ---------------------------------------------------------------------------
# Color picking
# ---------------------------------------------------------------------------

def _extract_caption_color(caption: str) -> str | None:
    """Return the first color-word hex found in *caption*, or None."""
    lower = caption.lower()
    for word, hex_color in COLOR_WORDS.items():
        if word in lower:
            return hex_color
    return None


def pick_color_for_brick(caption: str, category: str, z: int, seed: int) -> str:
    """Pick a deterministic color for a single brick.

    Priority:
    1. Color keyword in *caption* (deterministic, same for all bricks in structure).
    2. Category palette.
    3. Default palette.

    Ground bricks (``z == 0``) receive 2x weight for dark colors within
    whichever palette is selected.
    """
    rng = random.Random(seed)

    # 1. Caption color words — use as the sole candidate
    caption_color = _extract_caption_color(caption)
    if caption_color is not None:
        return caption_color

    # 2. Choose palette
    palette = CATEGORY_PALETTES.get(category, DEFAULT_PALETTE)

    # 3. Weighted selection: dark colors get double weight at ground level
    if z == 0:
        weights = [2 if c in _DARK_COLORS else 1 for c in palette]
    else:
        weights = [1] * len(palette)

    return rng.choices(palette, weights=weights, k=1)[0]


# ---------------------------------------------------------------------------
# Structure colorization
# ---------------------------------------------------------------------------

def colorize_structure(
    raw_bricks: list[tuple[int, int, int, int, int]],
    caption: str,
    category: str,
    seed: int,
) -> list[Brick]:
    """Assign colors to every brick in the structure.

    Each brick gets an independent seed derived from *seed* + its index so
    the results are fully deterministic regardless of list order.
    """
    result: list[Brick] = []
    for i, (h, w, x, y, z) in enumerate(raw_bricks):
        brick_seed = seed + i
        color = pick_color_for_brick(caption, category, z, brick_seed)
        result.append(Brick(h=h, w=w, x=x, y=y, z=z, color=color))
    return result


# ---------------------------------------------------------------------------
# Training example formatting
# ---------------------------------------------------------------------------

_SYSTEM_MSG = "You are a LEGO master builder."

_USER_TEMPLATE = (
    "Create a colored LEGO model. Format: <dims> (<x>,<y>,<z>) <#hex>.\n"
    "Allowed dims: 2x4, 4x2, 2x6, 6x2, 1x2, 2x1, 1x4, 4x1, 1x6, 6x1, 1x8, 8x1, 1x1, 2x2.\n"
    "All bricks are 1 unit tall.\n"
    "\n"
    "### Input:\n"
    "{caption}"
)


def format_training_example(caption: str, bricks: list[Brick]) -> dict:
    """Format a caption + colored brick list as a chat-style training example.

    Returns::

        {"messages": [
            {"role": "system",    "content": "..."},
            {"role": "user",      "content": "...{caption}"},
            {"role": "assistant", "content": "2x4 (5,3,0) #C91A09\\n..."},
        ]}
    """
    assistant_content = "\n".join(serialize_brick(b) for b in bricks)
    return {
        "messages": [
            {"role": "system", "content": _SYSTEM_MSG},
            {"role": "user",   "content": _USER_TEMPLATE.format(caption=caption)},
            {"role": "assistant", "content": assistant_content},
        ]
    }


# ---------------------------------------------------------------------------
# Main — dataset download + JSONL writing
# ---------------------------------------------------------------------------

# Helper only used inside main()'s dataset-download loop — excluded with main.
def _make_seed(structure_id: str, caption_index: int) -> int:  # pragma: no cover
    """Deterministic integer seed from structure_id + caption index."""
    key = f"{structure_id}:{caption_index}".encode()
    digest = hashlib.md5(key).digest()
    return int.from_bytes(digest[:4], "big")


# CLI entry point — downloads the HF dataset and writes JSONL; not run in tests.
def main() -> None:  # pragma: no cover
    """Download StableText2Brick and write JSONL training data.

    Output files:
        data/brick_training/train.jsonl
        data/brick_training/test.jsonl
    """
    from datasets import load_dataset  # type: ignore[import-untyped]

    out_dir = Path("data/brick_training")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading StableText2Brick from HuggingFace...", flush=True)
    ds = load_dataset("AvaLovelace/StableText2Brick")
    print(f"  Loaded: {ds}", flush=True)

    for split_name in ["train", "test"]:
        if split_name not in ds:
            print(f"  Skipping {split_name} (not in dataset)", flush=True)
            continue

        out_path = out_dir / f"{split_name}.jsonl"
        split = ds[split_name]
        total = len(split)
        count = 0
        errors = 0

        print(f"\nProcessing {split_name}: {total} structures × 5 captions...", flush=True)

        with out_path.open("w", encoding="utf-8") as fh:
            for row_idx, row in enumerate(split):
                structure_id: str = row["structure_id"]
                category_id: str = row["category_id"]
                category: str = SHAPENET_CATEGORIES.get(category_id, "")
                captions: list[str] = row["captions"]
                raw_text: str = row["bricks"]

                try:
                    raw_bricks = parse_st2b_bricks(raw_text)
                except Exception:
                    errors += 1
                    continue

                if not raw_bricks:
                    errors += 1
                    continue

                for cap_idx, caption in enumerate(captions):
                    if not caption:
                        continue
                    seed = _make_seed(structure_id, cap_idx)
                    bricks = colorize_structure(raw_bricks, caption, category, seed)
                    example = format_training_example(caption, bricks)
                    fh.write(json.dumps(example) + "\n")
                    count += 1

                if (row_idx + 1) % 5000 == 0 or row_idx + 1 == total:
                    print(f"  [{row_idx+1}/{total}] {count} examples, {errors} errors", flush=True)

        print(f"  {split_name}: {count} examples written → {out_path}", flush=True)
        if errors:
            print(f"  {split_name}: {errors} structures skipped due to errors", flush=True)


if __name__ == "__main__":
    main()
