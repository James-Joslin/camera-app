# TODO verification — 2026-09-19

Release: `v2026-09-14T212440Z`, status **experimental**.

## Completed implementation

- Live browser capture submits at most one JPEG per player at a time, with no frame queue. HTTP 429 drops the captured frame and samples a fresh one later.
- Confidence and start/stop controls, contain/meet box scaling, cancellation, and rejection of delayed or invalidated results are implemented.
- Webtop Storage Explorer is attached to Camera Azurite and lists `computer-vision-data` and `computer-vision-models`. The saved connection and successful container listing were verified after container recreation.
- Netron rendered 1,646 nodes from the exported local INT8 graph. Both inspection tools return HTTP 200 on their documented loopback ports.

## Checks

- Chromium regression: JPEG uploads, serial requests, threshold changes, square viewport letterboxing, HTTP 429 fresh-frame recovery, delayed response rejection, stop, pause, hidden tab, seek, stalled video and unmount.
- Real Chromium HLS playback/canvas/JPEG/inference through four replay players.
- Production frontend build, Prettier and proxy/cancellation regression passed.
- Backend: 9 tests passed in an isolated SDK container.
- Python: 5 reference-runtime tests and 2 load-harness tests passed in a disposable training container. Host Python lacks pytest; read-only test mounts produced only pytest cache warnings.
- Development/production Compose, tools overlay, Docker build checks and shell syntax passed.

## Replay measurements

Hardware: Intel Core i3-13100 (4 cores / 8 threads); `OPENVINO_THREADS=1`, `INFERENCE_REQUESTS=1`, `INFERENCE_QUEUE_LIMIT=2`. Training was absent and other services were idle. Host/model snapshots are saved with the reports.

Input: one 20.033-second H.264 1920×1080 camera clip, sampled into 20 JPEGs. The sampled scene is an indoor room with no visible person. This is a smoke dataset, not representative production validation.

| HTTP path, four workers at 2 FPS each | Successful / attempted | HTTP p95 | Warm inference p95 |
| --- | ---: | ---: | ---: |
| direct | 120 / 160 | 83.6 ms | 23.6 ms |
| frontend | 120 / 160 | 94.1 ms | 18.3 ms |

Each HTTP route returned 40 busy responses (HTTP 429) during the synchronized four-worker load. Success latency excludes rejected requests; the JSON contains all response counts, per-camera results and all-request timing. Encoded-frame HTTP timing excludes capture/encoding and HLS age.

| Browser replay player | Successful measured responses | Capture-to-response p95 | Encoding p95 |
| --- | ---: | ---: | ---: |
| Replay 1 | 51 | 70.9 ms | 37.3 ms |
| Replay 2 | 50 | 76.7 ms | 19.2 ms |
| Replay 3 | 50 | 60.4 ms | 24.8 ms |
| Replay 4 | 50 | 59.4 ms | 21.9 ms |

The browser run excludes the first five responses per player and includes real JPEG encoding and the frontend proxy. Each player samples again 500 ms after completing its last request, so this load differs from the paced synchronized HTTP harness. Browser timing starts at canvas capture and ends at parsed response; it excludes camera-to-HLS delay and overlay paint.

Local evidence (ignored to keep footage and generated output out of Git):

- `model_training/output/camera-validation/2026-09-19-smoke/clip.mp4` and `frames/`
- `capture.json`, `load-before.txt`, `inference-status.json`
- `http-4-camera-replay.json`, `browser-benchmark.json`

## Still required

- Representative, independently reviewed person/ignore/crowd labels across deployment cameras, lighting, distance and occlusion, then evaluation of the exact model and the camera-domain gate.
- Intended deployment camera count/sample rate and representative footage under that load, plus measured camera-to-display latency. A replay of one indoor scene does not prove these requirements.

The two corresponding TODOs remain open. No model release status or publication pointer was changed. The temporary capture account was deleted and its stream stopped. See [the repeatable validation and benchmark commands](camera-validation.md).
