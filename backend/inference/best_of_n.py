# backend/inference/best_of_n.py
from typing import Iterable

def rank_candidates(candidates: Iterable[dict]) -> list[dict]:
    """Sort candidates descending by (stable, brick_count).

    Preserves input order as the final tiebreaker so ranking is deterministic
    when two candidates are otherwise equal — important for the eval plot.
    """
    indexed = list(enumerate(candidates))
    indexed.sort(
        key=lambda pair: (
            1 if pair[1]["stable"] else 0,
            pair[1]["brick_count"],
            -pair[0],  # earlier index wins ties
        ),
        reverse=True,
    )
    return [c for _, c in indexed]
