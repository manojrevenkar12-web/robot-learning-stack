# Release check: PASS

Candidate `runs/act_main/best` (run `act_main`, dataset `v-8d8fc45d7b`), mode `full`.

| check | value | threshold | result |
|---|---|---|---|
| data.drop_rate_max_abs_error | 0.0008 | < 0.02 | pass |
| data.corrupt_frames_missed | 0 | == 0 | pass |
| data.wrist_clock_offset_error_ms | 8.0 | < 25 ms | pass |
| data.healthy_episode_fraction | 0.833 | reported (realistic profile fails ~13% of episodes by design) | pass |
| export.onnx_max_abs_diff | 6.020069122314453e-06 | < 1e-4 | pass |
| export.onnx_p95_ms | 33.5 | < 50 ms (15 Hz control budget is 66 ms) | pass |
| closed_loop.in_distribution | 0.433 | baseline 13/30; fail if drop >= 0.10 and p < 0.05 (drop +0.00, p 0.500) | pass |
| closed_loop.position_ood | 0.5 | baseline 15/30; fail if drop >= 0.10 and p < 0.05 (drop +0.00, p 0.500) | pass |
