# C# application inference

Application inference is served by the ASP.NET API at `/api/inference/detect`.
Training, calibration and export remain in Python. The framework-free Python
oracle in `api/tests/reference_runtime.py` exists only for numerical regression
tests; there is no Python HTTP inference service.

## Runtime

C# owns multipart handling, pooled upload buffers, bounded request admission,
worker scheduling, JSON responses and model lifecycle. A small checked-in C ABI
bridge owns OpenCV decode/resize and OpenVINO execution. This is a native C++
processing backend hosted in C#, not a managed reimplementation of OpenCV or
OpenVINO. Production does not install or execute Python. The build stage uses
the pinned OpenVINO 2025.3.0 wheel to obtain its native libraries and headers.
OpenCV comes from Debian Bookworm (4.6); the Python reference may use a different
OpenCV build, so compare decoded pixels and interpolation results as well as boxes.

Each worker reuses its resized Mat, candidate vectors and inference
request. Decoding allocates a fresh image to avoid stale-frame reuse on decoder
failure. Preprocessing resizes BGR first, then writes RGB float NCHW data directly
into the request's input tensor. It preserves round-to-even letterbox sizing,
symmetric zero padding before normalization, ImageNet means/stds, sigmoid score
filtering, top-K before valid-box filtering, IoU NMS at 0.45 and at most 100 boxes.
Boxes are mapped to the original image and truncated to integers, matching the
previous JSON response. Equal scores use ascending location index deterministically;
the old NumPy implementation did not define this tie ordering.

The supported model is the current `clean_ltrb` exported detector: one FP32 NCHW
landscape batch-one input, one person logit per location, and a named
`boxes_xyxy_pixels` output. FP32, compressed FP16 and INT8 XML/BIN pairs are accepted;
FP16 weight compression does not imply native CPU FP16 execution. Historical
anchor-offset graphs are rejected explicitly. V3 and V4 backbones both use this
same current contract.

## Deployment and configuration

From the repository root, rebuild the API and recreate API/frontend with the same
Compose file set normally used for the application. For the default development
stack:

```bash
docker compose build api
docker compose up -d --no-deps api frontend
```

Do not restart the training service for this change. Both API images build the
native bridge. Native libraries are stored outside the development source mount.
The existing `output/active-models` directory is mounted read-only at `/models`.

| Environment | Default | Meaning |
| --- | --- | --- |
| `MODEL_PATH` | `/models/person_detector_int8.xml` | Explicit XML path; matching BIN required |
| `OPENVINO_THREADS` | `1` | CPU inference threads, per compiled model |
| `INFERENCE_REQUESTS` | `1` | Independent reusable inference requests |
| `INFERENCE_QUEUE_LIMIT` | `2` | Additional admitted requests allowed to wait |
| `INFERENCE_MAX_UPLOAD_BYTES` | `20971520` | Maximum encoded image size |
| `MODEL_PRE_NMS_TOPK` | `1000` | Candidate cap before NMS |
| `INFERENCE_API_URL` | `http://api:8080` | Frontend upstream |

A full queue returns HTTP 429 before multipart parsing. Clients should discard
stale video frames rather than accumulate retries. Cancellation releases waiting
slots and buffers; a native inference already executing completes before its
worker is returned. ASP.NET's multipart and server request-size limits also apply.

`/api/inference/status` retains ready/loaded/model/device/error and adds backend
and concurrency information. Readiness before first use reports file availability;
first inference compiles and validates the graph. Model updates are manual: use **Settings → Refresh model** while signed in.
The frontend calls authenticated `POST /api/inference/refresh`. There is no
periodic polling or startup check against Azurite.
It downloads the selected XML/BIN pair, checks manifest sizes and SHA-256 hashes,
and compiles and runs a test image through a separate candidate before draining active requests and switching
workers. Failed updates leave the existing model serving. Inference responses
include `releaseId`; status includes `releaseId`, `releaseStatus`, and `refresh`.

Compose mounts a persistent `model-cache` volume at `/app/model-cache`. The last
successfully activated release is checksum-checked and restored locally on startup
without contacting Azurite. The read-only `/models` mount remains the initial
fallback. Cache cleanup retains the active release. Publication should upload
all artifacts and the manifest before updating `current.json`.

| Environment | Default | Meaning |
| --- | --- | --- |
| `MODEL_VARIANT` | `int8` | Published variant: int8, fp16, or fp32 |
| `MODEL_CACHE_DIR` | `/app/model-cache` | Writable release cache |
| `AZURITE_MODEL_CONTAINER` | `computer-vision-models` | Release blob container |
| `AZURITE_MODEL_CURRENT_POINTER` | `person_detector_ssd/current.json` | Published release pointer |
| `AZURITE_CONNECTION_STRING` | none | Blob storage connection |

Accepted experimental releases remain supported; the release status is displayed
in the frontend Settings and Activity views. "Latest" means the release selected by the pointer,
so publishing a previous accepted release also supports rollback. Refresh errors
are separate from inference readiness. Storage credentials stay in the API.

Deploy this feature once by rebuilding/recreating the API with the usual Compose
configuration; subsequent model updates use the Settings button and require no API/container restart.
Concurrent refresh attempts return HTTP 409. Refresh status reports manual mode,
progress, the last check, the last activation, and any error.

## Timings and verification

The response preserves `model`, `image`, `inferenceMs`, and `detections` and adds
`timings`: queue, initialization, multipart/read, decode, preprocessing, native
inference, postprocessing and total processing milliseconds. `processingMs`
excludes response serialization and network transfer. Unlike the Python runtime's
old measurement, `inferenceMs` excludes waiting for the inference lock. Measure
client HTTP duration for a full request comparison.

`tests/generate_inference_fixtures.py` generates input-dependent OpenVINO graphs
and reference detections using `api/tests/reference_runtime.py`. It requires the
training Python environment. `tests/verify_inference.py` checks landscape,
portrait and odd dimensions; JPEG/PNG decoding; multiple confidence thresholds;
NMS, clipping, output coordinates; invalid input; and concurrent requests.

```bash
python api/tests/generate_inference_fixtures.py /tmp/inference-fixtures api/tests/reference_runtime.py
# Start the validation API with /tmp/inference-fixtures mounted at /models and
# MODEL_PATH=/models/person_detector_fp32.xml (then repeat for fp16 and int8).
python api/tests/verify_inference.py /tmp/inference-fixtures http://localhost:8080 fp32
python api/tests/benchmark_inference.py \
  --url csharp=http://localhost:8080 \
  --images frame1.jpg frame2.png --cameras 4 --fps 2 \
  --hardware-notes "CPU model; idle; OpenVINO threads/requests settings" \
  --output /tmp/inference-benchmark.json
```

When comparing against another HTTP implementation, use the same exported model
and CPU configuration. The test-only Python oracle uses OpenVINO's automatic
CPU thread selection; it is for correctness, not a matched timing benchmark. Run on idle
hardware, with 8 warmups and 100 measured requests, then repeat under representative
multi-camera load. The fixture suite establishes correctness, not pedestrian AP
or an application speedup. Validate the trained FP32/INT8 exports against the full
500-image evaluator when assessing trained-model accuracy. Avoid drawing performance conclusions
from validation performed while model training is consuming CPU resources.

INT8 contract fixtures use synthetic calibration and are never promoted into
active-models. They establish wrapper correctness, not INT8 detection accuracy.
