"""Generate simple Best-of-N benchmark plots from bon_summary.csv.

The plotter intentionally uses only the Python standard library so benchmark
artifacts can be regenerated on GPU boxes without installing extra packages.
It writes SVG line charts under <run-dir>/plots/.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _format_tick(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _line_chart_svg(
    points: list[tuple[float, float]],
    *,
    title: str,
    x_label: str,
    y_label: str,
    y_min: float | None = None,
    y_max: float | None = None,
    width: int = 720,
    height: int = 420,
) -> str:
    margin_left = 72
    margin_right = 28
    margin_top = 48
    margin_bottom = 64
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    if not points:
        points = [(0.0, 0.0)]

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys) if y_min is None else y_min
    max_y = max(ys) if y_max is None else y_max
    if min_x == max_x:
        min_x -= 1
        max_x += 1
    if min_y == max_y:
        pad = 1.0 if max_y == 0 else abs(max_y) * 0.1
        min_y -= pad
        max_y += pad
    if y_min is None:
        min_y = min(0.0, min_y)
    if y_max is None:
        max_y = max_y * 1.08 if max_y > 0 else 1.0

    def sx(x: float) -> float:
        return margin_left + ((x - min_x) / (max_x - min_x)) * plot_w

    def sy(y: float) -> float:
        return margin_top + (1.0 - ((y - min_y) / (max_y - min_y))) * plot_h

    polyline = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in points)
    point_nodes = "\n".join(
        f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="4" fill="#111111" />'
        f'<text x="{sx(x):.2f}" y="{sy(y) - 10:.2f}" text-anchor="middle" '
        f'font-size="12">{_format_tick(y)}</text>'
        for x, y in points
    )

    x_ticks = sorted({int(x) if float(x).is_integer() else x for x in xs})
    x_tick_nodes = "\n".join(
        f'<line x1="{sx(float(x)):.2f}" y1="{margin_top + plot_h}" '
        f'x2="{sx(float(x)):.2f}" y2="{margin_top + plot_h + 6}" stroke="#333" />'
        f'<text x="{sx(float(x)):.2f}" y="{margin_top + plot_h + 24}" '
        f'text-anchor="middle" font-size="12">{x}</text>'
        for x in x_ticks
    )

    y_ticks = [min_y + (max_y - min_y) * i / 4 for i in range(5)]
    y_tick_nodes = "\n".join(
        f'<line x1="{margin_left - 6}" y1="{sy(y):.2f}" x2="{margin_left}" '
        f'y2="{sy(y):.2f}" stroke="#333" />'
        f'<line x1="{margin_left}" y1="{sy(y):.2f}" x2="{margin_left + plot_w}" '
        f'y2="{sy(y):.2f}" stroke="#dddddd" stroke-dasharray="2,4" />'
        f'<text x="{margin_left - 10}" y="{sy(y) + 4:.2f}" text-anchor="end" '
        f'font-size="12">{_format_tick(y)}</text>'
        for y in y_ticks
    )

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#ffffff" />
<text x="{width / 2:.0f}" y="28" text-anchor="middle" font-family="Arial, sans-serif" font-size="20" font-weight="700">{title}</text>
<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top + plot_h}" stroke="#111111" />
<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="#111111" />
{y_tick_nodes}
{x_tick_nodes}
<polyline fill="none" stroke="#0b6bcb" stroke-width="3" points="{polyline}" />
{point_nodes}
<text x="{margin_left + plot_w / 2:.0f}" y="{height - 18}" text-anchor="middle" font-family="Arial, sans-serif" font-size="14">{x_label}</text>
<text transform="translate(18 {margin_top + plot_h / 2:.0f}) rotate(-90)" text-anchor="middle" font-family="Arial, sans-serif" font-size="14">{y_label}</text>
</svg>
"""


def _read_bon_summary(run_dir: Path) -> list[dict[str, str]]:
    path = run_dir / "bon_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing Best-of-N summary: {path}")
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def generate_plots(run_dir: Path | str) -> list[Path]:
    """Generate all BoN plots for *run_dir* and return written paths."""
    run_dir = Path(run_dir)
    rows = _read_bon_summary(run_dir)
    rows = [row for row in rows if row.get("n")]
    rows.sort(key=lambda row: _as_float(row.get("n")))

    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        note = plot_dir / "README.md"
        note.write_text("No Best-of-N summary rows were available to plot.\n", encoding="utf-8")
        return [note]

    xys = {
        "stable_rate_vs_n.svg": (
            [(float(row["n"]), _as_float(row.get("stable_rate"))) for row in rows],
            "Best-of-N Stable Rate",
            "n",
            "candidate stable rate",
            0.0,
            1.0,
        ),
        "brick_count_vs_n.svg": (
            [(float(row["n"]), _as_float(row.get("avg_brick_count"))) for row in rows],
            "Best-of-N Brick Count",
            "n",
            "average picked brick count",
            0.0,
            None,
        ),
        "latency_vs_n.svg": (
            [(float(row["n"]), _as_float(row.get("avg_wall_time_ms"))) for row in rows],
            "Best-of-N Latency",
            "n",
            "average wall time (ms)",
            0.0,
            None,
        ),
    }

    written: list[Path] = []
    for filename, (points, title, x_label, y_label, y_min, y_max) in xys.items():
        out = plot_dir / filename
        out.write_text(
            _line_chart_svg(
                points,
                title=title,
                x_label=x_label,
                y_label=y_label,
                y_min=y_min,
                y_max=y_max,
            ),
            encoding="utf-8",
        )
        written.append(out)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot LEGOGen benchmark results.")
    parser.add_argument("run_dir", type=Path, help="benchmark_runs/<timestamp> directory")
    args = parser.parse_args(argv)
    written = generate_plots(args.run_dir)
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
