"""Build stability and legality checker for LEGO descriptions."""

import csv
import gzip
import json
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.config import (
    CACHE_DIR,
    QUANTITY_WARN_THRESHOLD,
    QUANTITY_FAIL_THRESHOLD,
    SUPPORT_RATIO_WARN,
    TOP_HEAVY_RATIO,
    MIN_CANTILEVER_CONNECTIONS,
)
from backend.inference.constraint_engine import POSITION_ORDER

# ── Data structures ───────────────────────────────────────────────────

POSITION_TIER = POSITION_ORDER  # alias for backward compat

SIDE_POSITIONS = {"left", "right", "front", "back"}


@dataclass
class CheckResult:
    name: str
    category: str  # "legality" | "stability"
    status: str  # "pass" | "warn" | "fail"
    message: str
    details: dict | None = None


@dataclass
class ValidationReport:
    score: int
    checks: list[CheckResult] = field(default_factory=list)
    summary: str = ""


# ── Catalog singletons ────────────────────────────────────────────────

_known_parts: frozenset[str] | None = None
_known_colors: set[str] | None = None


def _load_known_parts() -> frozenset[str]:
    global _known_parts
    if _known_parts is not None:
        return _known_parts
    parts_path = CACHE_DIR / "parts.csv.gz"
    part_nums: set[str] = set()
    if parts_path.exists():
        with gzip.open(parts_path, "rt") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if row:
                    part_nums.add(row[0])
    _known_parts = frozenset(part_nums)
    return _known_parts


def _load_known_colors() -> set[str]:
    global _known_colors
    if _known_colors is not None:
        return _known_colors
    colors_path = CACHE_DIR / "colors.json"
    names: set[str] = set()
    if colors_path.exists():
        with open(colors_path) as f:
            data = json.load(f)
        for entry in data.values():
            name = entry.get("name", "")
            if name:
                names.add(name.lower())
    _known_colors = names
    return _known_colors


# ── Helper ────────────────────────────────────────────────────────────

def _part_count(sa: dict) -> int:
    """Sum of quantities across all parts in a subassembly."""
    return sum(p.get("quantity", 1) for p in sa.get("parts", []))


# ── StabilityChecker ──────────────────────────────────────────────────

class StabilityChecker:
    """Runs legality and stability checks on a LEGO build description."""

    # ── Legality checks ───────────────────────────────────────────────

    def check_part_existence(self, desc: dict) -> CheckResult:
        known = _load_known_parts()
        if not known:
            return CheckResult(
                name="part_existence",
                category="legality",
                status="warn",
                message="Parts catalog not available — skipped check.",
            )
        unknown: list[str] = []
        for sa in desc.get("subassemblies", []):
            for part in sa.get("parts", []):
                pid = part.get("part_id", "")
                if pid and pid not in known:
                    unknown.append(pid)
        if not unknown:
            return CheckResult(
                name="part_existence",
                category="legality",
                status="pass",
                message="All part IDs found in the LEGO catalog.",
            )
        unique = list(dict.fromkeys(unknown))
        status = "fail" if len(unique) > 5 else "warn"
        return CheckResult(
            name="part_existence",
            category="legality",
            status=status,
            message=f"{len(unique)} unknown part ID(s): {', '.join(unique[:10])}.",
            details={"unknown_ids": unique},
        )

    def check_part_compatibility(self, desc: dict) -> CheckResult:
        mismatches: list[str] = []
        for sa in desc.get("subassemblies", []):
            sa_type = sa.get("type", "").lower()
            for part in sa.get("parts", []):
                cat = part.get("category", "").lower()
                if sa_type and cat and sa_type != cat:
                    mismatches.append(
                        f"{sa.get('name')}: part '{part.get('name')}' "
                        f"(category '{part.get('category')}') in subassembly type '{sa.get('type')}'"
                    )
        if not mismatches:
            return CheckResult(
                name="part_compatibility",
                category="legality",
                status="pass",
                message="All parts match their subassembly types.",
            )
        return CheckResult(
            name="part_compatibility",
            category="legality",
            status="warn",
            message=f"{len(mismatches)} part/subassembly type mismatch(es).",
            details={"mismatches": mismatches[:10]},
        )

    def check_color_validity(self, desc: dict) -> CheckResult:
        known = _load_known_colors()
        if not known:
            return CheckResult(
                name="color_validity",
                category="legality",
                status="warn",
                message="Color catalog not available — skipped check.",
            )
        invalid: list[str] = []
        for sa in desc.get("subassemblies", []):
            for part in sa.get("parts", []):
                color = part.get("color", "")
                if color and color.lower() not in known:
                    invalid.append(color)
        if not invalid:
            return CheckResult(
                name="color_validity",
                category="legality",
                status="pass",
                message="All colors are valid LEGO colors.",
            )
        unique = list(dict.fromkeys(invalid))
        return CheckResult(
            name="color_validity",
            category="legality",
            status="warn",
            message=f"{len(unique)} unrecognized color(s): {', '.join(unique[:10])}.",
            details={"invalid_colors": unique},
        )

    def check_quantity_reasonableness(self, desc: dict) -> CheckResult:
        issues: list[str] = []
        worst = "pass"
        for sa in desc.get("subassemblies", []):
            for part in sa.get("parts", []):
                qty = part.get("quantity", 0)
                if qty > QUANTITY_FAIL_THRESHOLD:
                    issues.append(f"{part.get('name')} x{qty}")
                    worst = "fail"
                elif qty > QUANTITY_WARN_THRESHOLD:
                    issues.append(f"{part.get('name')} x{qty}")
                    if worst != "fail":
                        worst = "warn"
        if worst == "pass":
            return CheckResult(
                name="quantity_reasonableness",
                category="legality",
                status="pass",
                message="All part quantities are reasonable.",
            )
        return CheckResult(
            name="quantity_reasonableness",
            category="legality",
            status=worst,
            message=f"{len(issues)} part(s) with unusual quantities: {', '.join(issues[:5])}.",
            details={"issues": issues},
        )

    # ── Stability checks ──────────────────────────────────────────────

    def check_foundation(self, desc: dict) -> CheckResult:
        for sa in desc.get("subassemblies", []):
            pos = sa.get("spatial", {}).get("position", "")
            if pos == "bottom":
                return CheckResult(
                    name="foundation",
                    category="stability",
                    status="pass",
                    message="Build has a bottom foundation subassembly.",
                )
        return CheckResult(
            name="foundation",
            category="stability",
            status="fail",
            message="No subassembly with position 'bottom' — build has no foundation.",
        )

    def check_connectivity(self, desc: dict) -> CheckResult:
        subs = desc.get("subassemblies", [])
        if len(subs) <= 1:
            return CheckResult(
                name="connectivity",
                category="stability",
                status="pass",
                message="Single subassembly — trivially connected.",
            )
        # Build undirected adjacency
        names = [sa.get("name", f"unnamed_{i}") for i, sa in enumerate(subs)]
        name_set = set(names)
        adj: dict[str, set[str]] = {n: set() for n in names}
        for sa in subs:
            n = sa.get("name", "")
            for target in sa.get("spatial", {}).get("connects_to", []):
                if target in name_set:
                    adj[n].add(target)
                    adj[target].add(n)

        # BFS from first node
        visited: set[str] = set()
        queue = deque([names[0]])
        visited.add(names[0])
        while queue:
            node = queue.popleft()
            for neighbor in adj[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        disconnected = name_set - visited
        if not disconnected:
            return CheckResult(
                name="connectivity",
                category="stability",
                status="pass",
                message="All subassemblies are connected.",
            )
        return CheckResult(
            name="connectivity",
            category="stability",
            status="fail",
            message=f"{len(disconnected)} disconnected subassembly(ies): {', '.join(sorted(disconnected))}.",
            details={"disconnected": sorted(disconnected)},
        )

    def check_support_ratio(self, desc: dict) -> CheckResult:
        subs = desc.get("subassemblies", [])
        sa_map = {sa.get("name", ""): sa for sa in subs}
        issues: list[str] = []
        for sa in subs:
            pos = sa.get("spatial", {}).get("position", "center")
            tier = POSITION_TIER.get(pos, 1)
            if tier == 0:
                continue  # bottom doesn't need support
            upper_count = _part_count(sa)
            connects = sa.get("spatial", {}).get("connects_to", [])
            support_count = 0
            for target_name in connects:
                target = sa_map.get(target_name)
                if not target:
                    continue
                target_tier = POSITION_TIER.get(
                    target.get("spatial", {}).get("position", "center"), 1
                )
                if target_tier < tier:
                    support_count += _part_count(target)
            if support_count > 0 and upper_count > SUPPORT_RATIO_WARN * support_count:
                issues.append(
                    f"'{sa.get('name')}' has {upper_count} parts but only {support_count} in lower supports"
                )
        if not issues:
            return CheckResult(
                name="support_ratio",
                category="stability",
                status="pass",
                message="All subassemblies have adequate lower support.",
            )
        return CheckResult(
            name="support_ratio",
            category="stability",
            status="warn",
            message=f"{len(issues)} subassembly(ies) may lack support: {issues[0]}.",
            details={"issues": issues},
        )

    def check_build_order(self, desc: dict) -> CheckResult:
        subs = desc.get("subassemblies", [])
        sa_map = {sa.get("name", ""): sa for sa in subs}
        issues: list[str] = []
        built: set[str] = set()
        for sa in subs:
            name = sa.get("name", "")
            tier = POSITION_TIER.get(
                sa.get("spatial", {}).get("position", "center"), 1
            )
            for target_name in sa.get("spatial", {}).get("connects_to", []):
                target = sa_map.get(target_name)
                if not target:
                    continue
                target_tier = POSITION_TIER.get(
                    target.get("spatial", {}).get("position", "center"), 1
                )
                # Only flag if target is at a strictly lower tier and hasn't been built yet;
                # same-tier connections are undirected and not a dependency.
                if target_tier < tier and target_name not in built:
                    issues.append(
                        f"'{name}' connects to '{target_name}' which hasn't been built yet"
                    )
            built.add(name)
        if not issues:
            return CheckResult(
                name="build_order",
                category="stability",
                status="pass",
                message="Build order is valid — all dependencies are met.",
            )
        return CheckResult(
            name="build_order",
            category="stability",
            status="warn",
            message=f"{len(issues)} build order issue(s): {issues[0]}.",
            details={"issues": issues},
        )

    def check_center_of_mass(self, desc: dict) -> CheckResult:
        bottom_parts = 0
        top_parts = 0
        for sa in desc.get("subassemblies", []):
            pos = sa.get("spatial", {}).get("position", "center")
            tier = POSITION_TIER.get(pos, 1)
            count = _part_count(sa)
            if tier == 0:
                bottom_parts += count
            elif tier == 3:
                top_parts += count
        if bottom_parts == 0 and top_parts == 0:
            return CheckResult(
                name="center_of_mass",
                category="stability",
                status="pass",
                message="No explicit top/bottom tiers to compare.",
            )
        if bottom_parts == 0 and top_parts > 0:
            return CheckResult(
                name="center_of_mass",
                category="stability",
                status="warn",
                message=f"Build has {top_parts} top-tier parts but no bottom-tier foundation parts.",
            )
        if top_parts > TOP_HEAVY_RATIO * bottom_parts:
            return CheckResult(
                name="center_of_mass",
                category="stability",
                status="warn",
                message=f"Top-heavy build: {top_parts} top parts vs {bottom_parts} bottom parts.",
                details={"top_parts": top_parts, "bottom_parts": bottom_parts},
            )
        return CheckResult(
            name="center_of_mass",
            category="stability",
            status="pass",
            message=f"Center of mass is balanced ({bottom_parts} bottom, {top_parts} top).",
        )

    def check_cantilever(self, desc: dict) -> CheckResult:
        issues: list[str] = []
        worst = "pass"
        for sa in desc.get("subassemblies", []):
            pos = sa.get("spatial", {}).get("position", "")
            if pos not in SIDE_POSITIONS:
                continue
            connections = len(sa.get("spatial", {}).get("connects_to", []))
            if connections == 0:
                issues.append(f"'{sa.get('name')}' at '{pos}' has no connections")
                worst = "fail"
            elif connections < MIN_CANTILEVER_CONNECTIONS:
                issues.append(
                    f"'{sa.get('name')}' at '{pos}' has only {connections} connection(s)"
                )
                if worst != "fail":
                    worst = "warn"
        if worst == "pass":
            return CheckResult(
                name="cantilever",
                category="stability",
                status="pass",
                message="No unsupported cantilever subassemblies.",
            )
        return CheckResult(
            name="cantilever",
            category="stability",
            status=worst,
            message=f"{len(issues)} cantilever issue(s): {issues[0]}.",
            details={"issues": issues},
        )

    # ── Main entry point ──────────────────────────────────────────────

    def validate(self, description: dict) -> ValidationReport:
        """Run all checks and return a scored report."""
        if not description or not description.get("subassemblies"):
            return ValidationReport(
                score=0,
                checks=[
                    CheckResult(
                        name="empty_build",
                        category="stability",
                        status="fail",
                        message="Build description is empty or has no subassemblies.",
                    )
                ],
                summary="0 passed, 0 warnings, 1 failure",
            )

        checks = [
            self.check_part_existence(description),
            self.check_part_compatibility(description),
            self.check_color_validity(description),
            self.check_quantity_reasonableness(description),
            self.check_foundation(description),
            self.check_connectivity(description),
            self.check_support_ratio(description),
            self.check_build_order(description),
            self.check_center_of_mass(description),
            self.check_cantilever(description),
        ]

        # Score
        score = 100
        passes = warns = fails = 0
        for c in checks:
            if c.status == "fail":
                score -= 15
                fails += 1
            elif c.status == "warn":
                score -= 5
                warns += 1
            else:
                passes += 1
        score = max(0, min(100, score))

        summary = f"{passes} passed, {warns} warning(s), {fails} failure(s)"

        return ValidationReport(score=score, checks=checks, summary=summary)
