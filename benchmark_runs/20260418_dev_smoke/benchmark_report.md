# LEGOGen Benchmark Report

- Run mode: `dev-mock`
- LEGOGEN_DEV: `1`
- Prompts: `2`
- Modes: `core, bon, stable-only, export`
- Git commit: `7463a65d43252d95919580ee7618424064670eed`
- Note: this is a dev/mock smoke result, not a real performance measurement.

## core

- run_mode: `dev-mock`
- prompt_count: `2`
- success_count: `2`
- success_rate: `1.0`
- stable_rate: `1.0`
- final_stable_rate: `1.0`
- parse_valid_rate: `1.0`
- collision_free_rate: `0.0`
- recomputed_stable_rate: `1.0`
- export_success_rate: `1.0`
- avg_brick_count: `12.0`
- avg_wall_time_ms: `0.017`
- p50_wall_time_ms: `0.017`
- avg_generation_time_ms: `1.0`
- avg_rejections: `0.0`
- avg_rollbacks: `0.0`

## bon

- n: `1`
- prompt_count: `2`
- success_count: `2`
- success_rate: `1.0`
- stable_rate: `1.0`
- picked_stable_rate: `1.0`
- final_stable_rate: `1.0`
- recomputed_stable_rate: `1.0`
- export_success_rate: `1.0`
- avg_brick_count: `12.0`
- avg_wall_time_ms: `0.021`
- p50_wall_time_ms: `0.021`
- avg_generation_time_ms: `1.0`

## stable-only

- run_mode: `dev-mock`
- prompt_count: `2`
- http_200_count: `2`
- http_422_count: `0`
- http_504_count: `0`
- other_status_count: `0`
- accepted_rate: `1.0`
- final_stable_rate_among_accepted: `1.0`
- parse_valid_rate_among_accepted: `1.0`
- export_success_rate_among_accepted: `1.0`
- avg_wall_time_ms: `7.131`

## export

- run_mode: `dev-mock`
- export_attempt_count: `2`
- export_success_count: `2`
- export_success_rate: `1.0`
- header_presence_rate: `1.0`
- part_line_count_match_rate: `1.0`
- avg_export_time_ms: `0.46`

## Plots

- `plots/stable_rate_vs_n.svg`
- `plots/brick_count_vs_n.svg`
- `plots/latency_vs_n.svg`
