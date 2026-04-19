# Stable-Only Route Benchmark Summary

Route-level require_stable=True results. This measures HTTP 200/422/504 behavior; it is not a retry-until-stable benchmark.

| run_mode | prompt_count | http_200_count | http_422_count | http_504_count | other_status_count | accepted_rate | final_stable_rate_among_accepted | parse_valid_rate_among_accepted | export_success_rate_among_accepted | avg_wall_time_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| real | 10 | 0 | 0 | 0 | 10 | 0.0 | 0.0 | 0.0 | 0.0 | 8.056 |
