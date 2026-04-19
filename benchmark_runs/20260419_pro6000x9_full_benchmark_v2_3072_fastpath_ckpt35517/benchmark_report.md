# LEGOGen Benchmark Report

- Run mode: `real`
- LEGOGEN_DEV: `0`
- Prompts: `10`
- Modes: `core, bon, stable-only, export`
- Git commit: `15de8052543890ac1417b93d848485cf0e0cfbd8`
- Comparison label: `qwen35-4b-brick-lora-v2-pro6000-9gpu-3072-fastpath-20260418_225200@checkpoint-35517`
- Brick model ID: `Qwen/Qwen3.5-4B`
- Brick checkpoint dir: `/root/autodl-tmp/Lego-Gen/backend/models/checkpoints/qwen35-4b-brick-lora-v2-pro6000-9gpu-3072-fastpath-20260418_225200`
- Visible CUDA devices: `9`
- Model devices: `cuda:8`
- Model parallel active: `False`
- Note: multiple CUDA devices were visible, but the loaded model did not shard across them.

## core

- run_mode: `real`
- quick_smoke: `False`
- capped_generation: `False`
- max_bricks_per_sample: `None`
- sample_timeout_s: `None`
- stability_check_interval: `None`
- prompt_count: `10`
- success_count: `0`
- success_rate: `0.0`
- stable_rate: `0.0`
- final_stable_rate: `0.0`
- parse_valid_rate: `0.0`
- collision_free_rate: `0.0`
- recomputed_stable_rate: `0.0`
- export_success_rate: `0.0`
- avg_brick_count: `0.0`
- avg_wall_time_ms: `0.0`
- p50_wall_time_ms: `0.0`
- avg_generation_time_ms: `0.0`
- avg_rejections: `0.0`
- avg_rollbacks: `0.0`
- hit_max_bricks_rate: `0.0`
- hit_max_seconds_rate: `0.0`

## bon

- n: `1`
- quick_smoke: `False`
- capped_generation: `False`
- max_bricks_per_sample: `None`
- sample_timeout_s: `None`
- stability_check_interval: `None`
- prompt_count: `10`
- success_count: `0`
- success_rate: `0.0`
- stable_rate: `0.0`
- picked_stable_rate: `0.0`
- final_stable_rate: `0.0`
- recomputed_stable_rate: `0.0`
- export_success_rate: `0.0`
- avg_brick_count: `0.0`
- avg_wall_time_ms: `0.0`
- p50_wall_time_ms: `0.0`
- avg_generation_time_ms: `0.0`
- hit_max_bricks_rate: `0.0`
- hit_max_seconds_rate: `0.0`

## stable-only

- run_mode: `real`
- prompt_count: `10`
- http_200_count: `0`
- http_422_count: `0`
- http_504_count: `0`
- other_status_count: `10`
- accepted_rate: `0.0`
- final_stable_rate_among_accepted: `0.0`
- parse_valid_rate_among_accepted: `0.0`
- export_success_rate_among_accepted: `0.0`
- avg_wall_time_ms: `8.056`

## export

- run_mode: `real`
- export_attempt_count: `0`
- export_success_count: `0`
- export_success_rate: `0.0`
- header_presence_rate: `0.0`
- part_line_count_match_rate: `0.0`
- avg_export_time_ms: `0.0`

## Plots

- `plots/stable_rate_vs_n.svg`
- `plots/brick_count_vs_n.svg`
- `plots/latency_vs_n.svg`
