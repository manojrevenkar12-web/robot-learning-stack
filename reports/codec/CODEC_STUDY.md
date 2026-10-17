# Video encoding study (front camera, 160x120, 12 episodes, 1361 frames)

| setting | bytes / frame | vs raw | PSNR mean (dB) | PSNR 1st pct (dB) | encode fps | decode fps | frame count exact |
|---|---|---|---|---|---|---|---|
| raw (uncompressed) | 57,600 | 1.0x | lossless | - | - | - | yes |
| JPEG quality 90 per frame (as in the bags) | 4,067 | 14.2x | 50.06 | 48.09 | 5264 | 3779 | yes |
| FFV1 lossless | 12,704 | 4.5x | 99.0 | 99.0 | 803 | 821 | yes |
| H.264 CRF 23 | 266 | 216.5x | 37.73 | 36.47 | 575 | 4222 | yes |
| H.264 CRF 30 | 122 | 470.8x | 34.68 | 33.4 | 751 | 4550 | yes |
| H.264 CRF 23, GOP 2 | 837 | 68.8x | 37.76 | 36.03 | 761 | 2713 | yes |
| HEVC CRF 28 | 172 | 334.0x | 35.8 | 33.51 | 413 | 3840 | yes |
| AV1 CRF 30, GOP 2 (release setting) | 829 | 69.5x | 38.77 | 36.58 | 487 | 4620 | yes |
| AV1 CRF 40, GOP 2 | 560 | 102.9x | 37.0 | 34.15 | 545 | 5036 | yes |

Reference frames are the staged frames, which were decoded from the bag's per-frame JPEGs; re-encoding them as JPEG at the
same quality is close to idempotent, so the JPEG row's PSNR is optimistic. GOP = keyframe interval: GOP 2 (LeRobot's
default) costs size but lets a training loader seek to any frame by decoding at most one extra frame; long-GOP H.264
is much smaller but random access has to decode from the previous keyframe.
