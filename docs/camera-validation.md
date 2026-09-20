# Camera validation and deployment benchmark

The release remains experimental until the camera-domain and project quality
gates pass. Synthetic frames, CityPersons alone, and model-generated boxes are
not camera-domain ground truth.

1. Use deployment cameras and record consented clips covering day/night,
   backlighting, empty scenes, near/far people, occlusion, movement, and each
   installation angle. Record camera, UTC interval, resolution, lighting and
   scene conditions. Keep clips under ignored `model_training/output/camera-validation/`.
2. Sample frames across independent clips. Manually annotate every person and
   ignore/crowd region, review the labels, and keep calibration and validation
   clips disjoint. Agree that the footage represents deployment conditions
   before treating results as production evidence.
3. Evaluate the exact exported model using the project's evaluator and preserve
   the image/label manifest, model SHA-256, evaluator options and metrics. Supply
   the resulting metric JSON to `scripts/optimize-model.sh --camera-metrics PATH`
   with `--release-status production`. The release workflow also requires the
   project accuracy and quantization gates; do not edit `current.json` by hand.

For a finite recording from a running camera, use FFmpeg against its HLS URL:

```bash
mkdir -p model_training/output/camera-validation
ffmpeg -i http://localhost:3101/streams/CAMERA_ID/index.m3u8 \
  -t 60 -map 0:v:0 -an -c:v copy \
  model_training/output/camera-validation/CAMERA_ID-DATE.mp4
```

Start the camera in the dashboard first and confirm the picture and playlist
are advancing. Stale HLS files do not prove a camera is connected.

Once training and other CPU-heavy work have finished, run the HTTP load harness
with representative frames at the intended camera count and sample rate:

```bash
python3 api/tests/benchmark_inference.py \
  --url direct=http://localhost:5255 --url frontend=http://localhost:3101 \
  --images model_training/output/camera-validation/frames/*.jpg \
  --cameras 4 --fps 2 --warmups 10 --iterations 120 \
  --hardware-notes 'CPU model; idle; OPENVINO_THREADS=1; INFERENCE_REQUESTS=1' \
  --output model_training/output/camera-validation/http-benchmark.json
```

The report includes warm inference, queue/processing/HTTP latency percentiles,
429 and other failure counts, missed capture slots, throughput and per-camera
statistics. One worker per camera bounds outstanding requests; missed capture
slots and HTTP 429 frames are discarded. Run several camera counts and preserve
`docker stats --no-stream`, CPU model and model release ID with each report.
HTTP measurements exclude HLS transport, browser decode/capture/encoding and
rendering. Measure these separately in the live dashboard before claiming
camera-to-display end-to-end latency. The browser detection status reports the
capture-to-response latency; HLS age must also be measured against a camera
clock or a visible synchronized clock in the scene.

## Browser regression checks

With the development frontend running, `./scripts/check-live-detection.sh`
builds an isolated Chromium test runner and verifies encoded JPEG submission,
serial requests, threshold changes, contain/meet overlay scaling, 429 fresh-frame
capture, stale-response rejection, and stop/pause behavior. It uses synthetic
decoded-video and inference fixtures; it is not an accuracy or latency benchmark.
`CAMERA_TEST_URL` overrides the default `http://localhost:3101`.

Measure real browser capture, JPEG encoding, proxy and inference from a running
HLS source with:

```bash
CAMERA_BENCH_STREAM_URL=/streams/CAMERA_ID/index.m3u8 \
CAMERA_BENCH_HARDWARE_NOTES='CPU model; model settings; idle; same source replayed across 4 players' \
./scripts/benchmark-browser.sh
```

This writes `/tmp/camera-browser-benchmark/browser-benchmark.json` (override
`CAMERA_BENCH_OUTPUT_DIR`). `CAMERA_BENCH_CAMERAS` and `CAMERA_BENCH_SECONDS`
control the load. It plays the actual HLS stream in Chromium and calls the real
inference endpoint; only dashboard camera metadata is replaced to create the
specified player count. The first five responses per player are excluded from
summaries. Raw samples include HTTP status, capture/encode/inference timings,
release ID and image dimensions. Replaying one camera across players tests load,
not camera diversity, and does not establish scene representativeness. Timing
ends at the parsed response, excluding camera-to-HLS delay and overlay painting.
