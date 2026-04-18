"""Prepare brick training data from StableText2Brick (HuggingFace dataset).

Downloads the dataset, assigns deterministic colors to monochrome bricks,
and writes JSONL training/test files for fine-tuning.

Usage (do NOT run during tests — downloads the dataset):
    python -m backend.data_pipeline.prepare_brick_dataset
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import random
from pathlib import Path

from backend.brick.parser import Brick, serialize_brick

# ---------------------------------------------------------------------------
# Color maps
# ---------------------------------------------------------------------------

COLOR_WORDS: dict[str, str] = {
    "dark gray": "6D6E5C",
    "dark grey": "6D6E5C",
    "light gray": "9BA19D",
    "light grey": "9BA19D",
    "red": "C91A09",
    "blue": "0055BF",
    "green": "237841",
    "yellow": "F2CD37",
    "black": "05131D",
    "white": "FFFFFF",
    "brown": "6D6E5C",
    "gray": "9BA19D",
    "grey": "9BA19D",
    "orange": "F2CD37",
    "pink": "FFFFFF",
    "purple": "0055BF",
    "tan": "FFFFFF",
}

DEFAULT_PALETTE: list[str] = [
    "C91A09", "0055BF", "237841", "F2CD37",
    "05131D", "FFFFFF", "6D6E5C", "9BA19D",
]

# Dark colors for ground-level weight boost
_DARK_COLORS: set[str] = {"05131D", "6D6E5C", "237841", "C91A09"}

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
    "basket":    ["6D6E5C", "FFFFFF", "9BA19D", "C91A09", "F2CD37"],
    "bed":       ["FFFFFF", "9BA19D", "C91A09", "0055BF", "6D6E5C"],
    "bench":     ["6D6E5C", "9BA19D", "05131D", "FFFFFF"],
    "birdhouse": ["6D6E5C", "FFFFFF", "F2CD37", "C91A09"],
    "bookshelf": ["6D6E5C", "9BA19D", "05131D", "FFFFFF"],
    "bottle":    ["237841", "0055BF", "C91A09", "9BA19D", "FFFFFF"],
    "bowl":      ["FFFFFF", "9BA19D", "C91A09", "0055BF", "F2CD37"],
    "bus":       ["F2CD37", "C91A09", "0055BF", "FFFFFF", "05131D"],
    "camera":    ["05131D", "9BA19D", "FFFFFF", "C91A09", "0055BF"],
    "car":       ["C91A09", "0055BF", "05131D", "FFFFFF", "9BA19D"],
    "chair":     ["6D6E5C", "9BA19D", "FFFFFF", "C91A09", "05131D"],
    "guitar":    ["6D6E5C", "FFFFFF", "9BA19D", "C91A09", "05131D"],
    "jar":       ["FFFFFF", "9BA19D", "C91A09", "237841", "0055BF"],
    "mug":       ["FFFFFF", "C91A09", "0055BF", "F2CD37", "9BA19D"],
    "piano":     ["05131D", "FFFFFF", "9BA19D", "6D6E5C"],
    "pot":       ["C91A09", "237841", "9BA19D", "F2CD37", "05131D"],
    "sofa":      ["C91A09", "0055BF", "9BA19D", "6D6E5C"],
    "table":     ["6D6E5C", "FFFFFF", "9BA19D", "05131D"],
    "tower":     ["9BA19D", "6D6E5C", "FFFFFF", "05131D"],
    "train":     ["C91A09", "05131D", "0055BF", "9BA19D", "F2CD37"],
    "vessel":    ["9BA19D", "0055BF", "FFFFFF", "C91A09", "05131D"],
}

_SAFE_COLOR_REMAP: dict[str, str] = {
    "FEC401": "F2CD37",
    "A0A5A9": "9BA19D",
    "958A73": "6D6E5C",
    "E4CD9E": "FFFFFF",
    "583927": "6D6E5C",
    "FE8A18": "F2CD37",
    "C870A0": "FFFFFF",
    "81007B": "0055BF",
}

_COLOR_WORD_RE = re.compile(
    r"\b(" + "|".join(re.escape(word) for word in sorted(COLOR_WORDS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

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

def _runtime_palette() -> set[str] | None:
    """Return runtime-allowed colors when available, otherwise None.

    Data generation can run before ``data/cache/colors.json`` exists. In that
    case we still emit the canonical safe palette above and let explicit audit
    mode decide whether the missing runtime palette should fail the job.
    """
    try:
        from backend.brick.constants import ALLOWED_COLORS

        return {color.upper() for color in ALLOWED_COLORS}
    except OSError:
        return None


def _extract_caption_color_mentions(caption: str) -> list[tuple[str, str]]:
    """Return ordered ``(word, hex)`` color mentions using word boundaries."""
    mentions: list[tuple[str, str]] = []
    seen_colors: set[str] = set()
    for match in _COLOR_WORD_RE.finditer(caption):
        word = match.group(1).lower()
        color = COLOR_WORDS[word]
        if color in seen_colors:
            continue
        mentions.append((word, color))
        seen_colors.add(color)
    return mentions


def _extract_caption_colors(caption: str) -> list[str]:
    """Return all distinct caption colors in text order."""
    return [color for _, color in _extract_caption_color_mentions(caption)]


def _extract_caption_color(caption: str) -> str | None:
    """Return the first caption color, preserving the old helper contract."""
    colors = _extract_caption_colors(caption)
    return colors[0] if colors else None


def _weighted_palette_choice(palette: list[str], z: int, rng: random.Random) -> str:
    if z == 0:
        weights = [2 if c in _DARK_COLORS else 1 for c in palette]
    else:
        weights = [1] * len(palette)
    return rng.choices(palette, weights=weights, k=1)[0]


def _mentions_any(caption: str, words: tuple[str, ...]) -> bool:
    lower = caption.lower()
    return any(re.search(rf"\b{re.escape(word)}\b", lower) for word in words)


def _component_color(caption: str, components: tuple[str, ...]) -> str | None:
    """Find colors tied to nearby component words such as ``yellow roof``."""
    color_alt = _COLOR_WORD_RE.pattern[len(r"\b("):-len(r")\b")]
    component_alt = "|".join(re.escape(word) for word in components)
    color_first = re.search(
        rf"\b({color_alt})\b(?:\s+\w+){{0,2}}\s+\b({component_alt})\b",
        caption,
        flags=re.IGNORECASE,
    )
    if color_first:
        return COLOR_WORDS[color_first.group(1).lower()]
    component_first = re.search(
        rf"\b({component_alt})\b(?:\s+\w+){{0,2}}\s+\b({color_alt})\b",
        caption,
        flags=re.IGNORECASE,
    )
    if component_first:
        return COLOR_WORDS[component_first.group(2).lower()]
    return None


def _first_non_black(colors: list[str]) -> str:
    for color in colors:
        if color != "05131D":
            return color
    return colors[0]


def pick_color_for_brick(
    caption: str,
    category: str,
    z: int,
    seed: int,
    *,
    max_z: int | None = None,
    brick_index: int = 0,
    h: int | None = None,
    w: int | None = None,
    x: int | None = None,
    y: int | None = None,
) -> str:
    """Pick a deterministic, palette-safe color for a single brick.

    Multi-color captions are interpreted as component colors instead of
    recoloring the entire structure with the first color word.
    """
    rng = random.Random(seed)
    palette = CATEGORY_PALETTES.get(category, DEFAULT_PALETTE)
    caption_colors = _extract_caption_colors(caption)
    top_z = z if max_z is None else max_z

    if not caption_colors:
        return _weighted_palette_choice(palette, z, rng)

    lower = caption.lower()
    is_house = category == "birdhouse" or _mentions_any(lower, ("house", "home", "cottage", "hut", "cabin"))
    is_vehicle = category in {"car", "bus", "train"} or _mentions_any(lower, ("car", "bus", "train", "truck", "vehicle"))
    is_tree = _mentions_any(lower, ("tree", "trunk", "leaf", "leaves"))
    is_tower = category == "tower" or _mentions_any(lower, ("tower", "pillar", "column"))

    if is_house:
        roof = _component_color(lower, ("roof", "rooftop", "top")) or (caption_colors[-1] if len(caption_colors) > 1 else None)
        wall = _component_color(lower, ("wall", "walls", "body", "side", "sides")) or (caption_colors[1] if len(caption_colors) > 1 else caption_colors[0])
        base = _component_color(lower, ("base", "ground", "floor")) or caption_colors[0]
        if z >= top_z and roof is not None:
            return roof
        if z == 0:
            return base
        return wall

    if is_vehicle:
        wheel = _component_color(lower, ("wheel", "wheels", "tire", "tires")) or ("05131D" if "05131D" in caption_colors else None)
        window = _component_color(lower, ("window", "windows", "windshield")) or ("FFFFFF" if "FFFFFF" in caption_colors else None)
        body = _component_color(lower, ("body", "hood", "roof", "side", "sides")) or _first_non_black(caption_colors)
        if z == 0 and wheel is not None and (brick_index % 3 == 0 or (h == 1 and w == 1)):
            return wheel
        if z >= top_z and window is not None:
            return window
        return body

    if is_tree:
        leaves = _component_color(lower, ("leaf", "leaves", "foliage", "top")) or ("237841" if "237841" in caption_colors else caption_colors[-1])
        trunk = _component_color(lower, ("trunk", "stem", "base")) or ("6D6E5C" if "6D6E5C" in DEFAULT_PALETTE else caption_colors[0])
        return trunk if z == 0 else leaves

    if is_tower and len(caption_colors) > 1:
        top = _component_color(lower, ("top", "cap", "roof")) or caption_colors[-1]
        base = _component_color(lower, ("base", "bottom")) or caption_colors[0]
        return top if z >= top_z else base

    if len(caption_colors) == 1:
        # Keep a single mentioned color dominant without making every support
        # brick identical across all categories.
        if z == 0 and rng.random() < 0.25:
            return _weighted_palette_choice(palette, z, rng)
        return caption_colors[0]

    return caption_colors[(z + brick_index) % len(caption_colors)]


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
    max_z = max((z for _, _, _, _, z in raw_bricks), default=0)
    for i, (h, w, x, y, z) in enumerate(raw_bricks):
        brick_seed = seed + i
        color = pick_color_for_brick(
            caption,
            category,
            z,
            brick_seed,
            max_z=max_z,
            brick_index=i,
            h=h,
            w=w,
            x=x,
            y=y,
        )
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
            {"role": "assistant", "content": "2x4 (5,3,0) #C91A09\\n...\\nDONE"},
        ]}
    """
    assistant_content = "\n".join([*(serialize_brick(b) for b in bricks), "DONE"])
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


def _training_palette_colors() -> set[str]:
    colors = set(DEFAULT_PALETTE)
    colors.update(COLOR_WORDS.values())
    for palette in CATEGORY_PALETTES.values():
        colors.update(palette)
    for color in _SAFE_COLOR_REMAP.values():
        colors.add(color)
    return {color.upper() for color in colors}


def audit_palette(mode: str = "warn") -> list[str]:
    """Check that generated training colors are accepted by the runtime palette."""
    if mode == "off":
        return []
    allowed = _runtime_palette()
    if allowed is None:
        message = "runtime palette unavailable; cannot audit training colors"
        if mode == "strict":
            raise RuntimeError(message)
        print(f"WARNING: {message}", flush=True)
        return []
    missing = sorted(color for color in _training_palette_colors() if color not in allowed)
    if missing:
        message = f"training colors not present in runtime palette: {', '.join(missing)}"
        if mode == "strict":
            raise RuntimeError(message)
        print(f"WARNING: {message}", flush=True)
    return missing


def _canary_structures() -> list[tuple[str, list[Brick]]]:
    """Small source-controlled v2 targets for color and DONE overfit checks."""
    return [
        (
            "a red house with white walls and a yellow roof",
            [
                Brick(2, 4, 0, 0, 0, "C91A09"),
                Brick(2, 4, 2, 0, 0, "C91A09"),
                Brick(2, 4, 0, 0, 1, "FFFFFF"),
                Brick(2, 4, 2, 0, 1, "FFFFFF"),
                Brick(2, 4, 0, 0, 2, "FFFFFF"),
                Brick(2, 4, 2, 0, 2, "FFFFFF"),
                Brick(2, 4, 0, 0, 3, "F2CD37"),
                Brick(2, 4, 2, 0, 3, "F2CD37"),
            ],
        ),
        (
            "a blue car with black wheels and white windows",
            [
                Brick(4, 2, 0, 0, 0, "05131D"),
                Brick(4, 2, 0, 2, 0, "05131D"),
                Brick(4, 2, 0, 0, 1, "0055BF"),
                Brick(4, 2, 0, 2, 1, "0055BF"),
                Brick(2, 2, 1, 1, 2, "FFFFFF"),
            ],
        ),
        (
            "a green tree with a dark gray trunk",
            [
                Brick(2, 2, 1, 1, 0, "6D6E5C"),
                Brick(2, 2, 1, 1, 1, "6D6E5C"),
                Brick(4, 2, 0, 1, 2, "237841"),
                Brick(2, 4, 1, 0, 3, "237841"),
                Brick(2, 2, 1, 1, 4, "237841"),
            ],
        ),
        (
            "a gray tower with a red top",
            [
                Brick(2, 2, 0, 0, 0, "9BA19D"),
                Brick(2, 2, 0, 0, 1, "9BA19D"),
                Brick(2, 2, 0, 0, 2, "9BA19D"),
                Brick(2, 2, 0, 0, 3, "C91A09"),
            ],
        ),
    ]


def canary_training_examples(repeat: int = 1) -> list[dict]:
    examples = [format_training_example(caption, bricks) for caption, bricks in _canary_structures()]
    return examples * repeat


def _write_jsonl_example(fh, example: dict) -> None:
    fh.write(json.dumps(example) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare LEGOGen Stage 2 brick training data.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/brick_training"))
    parser.add_argument("--include-canary", action="store_true", help="Append source-controlled v2 canary examples.")
    parser.add_argument("--canary-repeat", type=int, default=1, help="Train split repetition count for canary examples.")
    parser.add_argument(
        "--palette-audit",
        choices=["off", "warn", "strict"],
        default="warn",
        help="Verify generated training colors are accepted by the runtime palette.",
    )
    return parser


# CLI entry point — downloads the HF dataset and writes JSONL; not run in tests.
def main(argv: list[str] | None = None) -> None:  # pragma: no cover
    """Download StableText2Brick and write JSONL training data.

    Output files:
        data/brick_training/train.jsonl
        data/brick_training/test.jsonl
    """
    from datasets import load_dataset  # type: ignore[import-untyped]

    args = build_arg_parser().parse_args(argv)
    if args.canary_repeat <= 0:
        raise SystemExit("--canary-repeat must be positive")
    audit_palette(args.palette_audit)

    out_dir = args.output_dir
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
                    _write_jsonl_example(fh, example)
                    count += 1

                if (row_idx + 1) % 5000 == 0 or row_idx + 1 == total:
                    print(f"  [{row_idx+1}/{total}] {count} examples, {errors} errors", flush=True)

            if args.include_canary:
                repeat = args.canary_repeat if split_name == "train" else 1
                for example in canary_training_examples(repeat=repeat):
                    _write_jsonl_example(fh, example)
                    count += 1

        print(f"  {split_name}: {count} examples written → {out_path}", flush=True)
        if errors:
            print(f"  {split_name}: {errors} structures skipped due to errors", flush=True)


if __name__ == "__main__":
    main()
