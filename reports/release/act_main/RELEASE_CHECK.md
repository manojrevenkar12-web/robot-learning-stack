# Release check: FAIL

Candidate `runs/act_main/best` (run `act_main`, dataset `v-8d8fc45d7b`), mode `full`.

| check | value | threshold | result |
|---|---|---|---|
| data.drop_rate_max_abs_error | 0.211 | < 0.02 | **FAIL** |
| data.corrupt_frames_missed | 0 | == 0 | pass |
| data.wrist_clock_offset_error_ms | 7.7 | < 25 ms | pass |
| data.healthy_episode_fraction | 0.667 | >= 0.8 | **FAIL** |
| export.onnx_max_abs_diff | 6.020069122314453e-06 | < 1e-4 | pass |
| export.onnx_p95_ms | 40.6 | < 50 ms (15 Hz control budget is 66 ms) | pass |
| closed_loop.in_distribution | 0.433 | no baseline yet (recorded only) | pass |
| closed_loop.position_ood | 0.5 | no baseline yet (recorded only) | pass |
