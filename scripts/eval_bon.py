# scripts/eval_bon.py
"""Quality-vs-compute sweep: for each caption, run BoN at N in {1,2,4,8,16}
and write one CSV row per (caption, N) with (stable_rate, best_brick_count,
picked_index).

Usage:
    LEGOGEN_DEV=1 .venv/bin/python scripts/eval_bon.py \
        --captions data/brick_training/eval_captions.txt --out eval_bon.csv
"""
import argparse, csv, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.inference.brick_pipeline import get_brick_pipeline

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ns", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--strategy", choices=["rank", "cluster"], default="cluster")
    args = ap.parse_args()

    captions = [ln.strip() for ln in args.captions.read_text().splitlines() if ln.strip()]
    pipe = get_brick_pipeline()

    with args.out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["caption", "n", "stable_rate", "best_brick_count", "picked_index", "gen_ms"])
        for cap in captions:
            for n in args.ns:
                out = pipe.generate_best_of_n(cap, n=n, strategy=args.strategy)
                m = out["metadata"]
                w.writerow([cap, n, f"{m['stable_rate']:.3f}",
                            out["brick_count"], m["picked_index"],
                            m.get("generation_time_ms", 0)])

if __name__ == "__main__":
    main()
