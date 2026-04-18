# LEGOGen Benchmark Runbook

This runbook measures the current brick-coordinate pipeline only. It does not
revive the removed JSON path, `/api/validate`, or historical validator scores.
The suite writes raw rows, summaries, environment metadata, plots, and a short
`benchmark_report.md` into `benchmark_runs/<timestamp>/`.

## What It Measures

The benchmark suite has four modes:

| Mode | Path measured | Main files |
| --- | --- | --- |
| Core | `BrickPipeline.generate(prompt)` with `n=1` | `core_raw.jsonl`, `core_summary.csv`, `core_summary.md` |
| Best-of-N | `BrickPipeline.generate_best_of_n(prompt, n)` for `n in {1,2,4,8,16}` | `bon_raw.jsonl`, `bon_summary.csv`, `bon_summary.md`, `plots/*.svg` |
| Stable-only | `POST /api/generate-bricks` with `require_stable=true` through FastAPI `TestClient` | `stable_only_raw.jsonl`, `stable_only_summary.csv`, `stable_only_summary.md` |
| Export | `export_ldr()` for every successful core output | `export_raw.jsonl`, `export_summary.csv`, `export_summary.md` |

Stable-only mode records actual route outcomes: HTTP 200 for accepted stable
responses, HTTP 422 for unstable outputs rejected by `require_stable`, and HTTP
504 for route timeout. It is not a retry-until-stable benchmark.

## Prompt Set

The curated prompt set lives at:

```bash
benchmarks/prompts/core_prompts.txt
```

Blank lines and `#` comments are ignored. The run directory also stores the
exact prompts used as `prompts_used.txt`.

## Dev Smoke Run

Dev mode uses `MockBrickPipeline`. It is useful for checking the harness and
file outputs, but it is not a real performance result.

```bash
LEGOGEN_DEV=1 .venv/bin/python scripts/benchmark_legogen.py \
  --limit-prompts 2
```

The command prints the run directory. Inspect:

```bash
ls benchmark_runs/<timestamp>
sed -n '1,120p' benchmark_runs/<timestamp>/benchmark_report.md
```

## Real Run

Real benchmarking requires CUDA and the Stage 2 brick LoRA adapter at
`backend/models/checkpoints/qwen35-4b-brick-lora/adapter_config.json`.
The harness records Stage 1 checkpoint metadata too, but the benchmark itself
is text-only and does not run the image-to-caption model.

For a fast real-GPU sanity check, run the capped quick smoke first. It measures
validity and exportability, not full model performance:

```bash
LEGOGEN_DEV=0 .venv/bin/python scripts/benchmark_legogen.py \
  --quick-smoke
```

`--quick-smoke` is shorthand for core plus export on one prompt with
`--max-bricks-per-sample 24`, `--sample-timeout-s 90`, and
`--stability-check-interval 8`. Reports from capped runs are labeled as smoke
results and must not be compared to full uncapped performance numbers.

```bash
LEGOGEN_DEV=0 .venv/bin/python scripts/benchmark_legogen.py
```

If CUDA or the brick adapter is unavailable, the suite exits gracefully after
writing environment metadata, empty summaries, and a `benchmark_report.md` that
explains why the real run was skipped. To intentionally benchmark the base
brick model without the LoRA adapter, pass `--allow-base-model` and document
that choice in any report.

## Comparison Targets

The current harness benchmarks one configured Stage 2 target per run. The
supported comparison targets today are:

- the default Stage 2 LoRA checkpoint from `BRICK_CHECKPOINT_DIR`
- any alternate Stage 2 LoRA checkpoint by overriding `BRICK_CHECKPOINT_DIR`
- the base `Qwen/Qwen3.5-4B` model with `--allow-base-model`

There is not yet a built-in multi-model sweep in one command. The intended
workflow is one run per target, each with an explicit label:

```bash
BRICK_CHECKPOINT_DIR=/path/to/checkpoint-19000 \
LEGOGEN_DEV=0 .venv/bin/python scripts/benchmark_legogen.py \
  --comparison-label checkpoint-19000
```

The run directory records the comparison label, model ID, and checkpoint path
in both `benchmark_config.json` and `environment_metadata.json`.

## Multi-GPU Notes

The benchmark harness can run on multi-GPU machines because the inference path
loads the model with `device_map="auto"`. In practice this means:

- the harness works when multiple GPUs are visible
- `environment_metadata.json` records `CUDA_VISIBLE_DEVICES`, visible CUDA count,
  and the model's resolved device placement
- `benchmark_report.md` states whether model sharding across multiple GPUs was
  actually active for that run

For the current 4-bit Qwen 4B brick model, extra GPUs usually help with memory
headroom rather than throughput. Hugging Face's `device_map="auto"` big-model
inference can shard across GPUs, but it is still model-parallel inference, so
only one GPU is typically active at a time while layers are handed off. This
is still useful for larger comparison targets that do not fit comfortably on a
single GPU, but it should not be treated as guaranteed speedup.

## Common Extensions

The current suite focuses on validity, stability, route behavior, export, and
Best-of-N selection. Other common benchmarks worth running alongside it are:

- checkpoint-to-checkpoint A/B runs with the same prompt set and explicit labels
- prompt-faithfulness / canary quality evaluation via `scripts/eval_generation_quality.py`
- concurrency and throughput benchmarks at the HTTP route level
- VRAM and device-placement tracking for larger comparison targets

## Plotting

`scripts/benchmark_legogen.py` automatically creates Best-of-N SVG plots when
the BoN mode runs:

```text
plots/stable_rate_vs_n.svg
plots/brick_count_vs_n.svg
plots/latency_vs_n.svg
```

To regenerate plots from an existing run:

```bash
.venv/bin/python scripts/plot_benchmark_results.py benchmark_runs/<timestamp>
```

## Metrics

Core raw rows record:

```text
wall_time_ms, generation_time_ms, brick_count, stable, final_stable,
termination_reason, rejections, rollbacks, outlines_enabled,
palette_validation_enabled, requested_max_bricks, requested_max_seconds,
requested_stability_check_interval, hit_max_bricks, hit_max_seconds
```

The harness independently recomputes:

```text
parse_valid              via parse_brick_sequence
collision_free           via VoxelGrid.can_place in sequence order
recomputed_stable        via is_stable
export_success           via export_ldr
ldraw_header_present     header sanity check
ldraw_part_line_count    LDraw part-line count
```

Best-of-N summaries report per-`n` candidate stable rate from the existing
`generate_best_of_n` metadata, final stable rate for picked outputs, average
picked brick count, wall-clock latency, and export success rate.

## Method Notes

Always separate dev/mock and real results. `environment_metadata.json` records
`LEGOGEN_DEV`, Python, torch, CUDA, GPU names, Outlines availability, model IDs,
checkpoint presence, git commit, and dirty status.

Do not compare against historical numbers unless those raw artifacts are in the
run directory. This suite is intended to produce fresh, reproducible outputs for
the repository as it exists now.
