# Core Benchmark Summary

n=1 text generation through the current BrickPipeline.generate path. Capped runs are smoke tests, not full performance results.

| run_mode | quick_smoke | capped_generation | max_bricks_per_sample | sample_timeout_s | stability_check_interval | prompt_count | success_count | success_rate | stable_rate | final_stable_rate | parse_valid_rate | collision_free_rate | recomputed_stable_rate | export_success_rate | avg_brick_count | avg_wall_time_ms | p50_wall_time_ms | avg_generation_time_ms | avg_rejections | avg_rollbacks | hit_max_bricks_rate | hit_max_seconds_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| real | False | False | None | None | None | 10 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
