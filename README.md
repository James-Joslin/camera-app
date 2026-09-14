# Sentinel Camera Software

High — multi-camera inference is not implemented yet. C# currently produces stream-copy HLS only ([StreamManager.cs (line 80)](/home/james/dockerfiles/camera-software/api/Streaming/StreamManager.cs:80)); it does not extract or forward inference frames. Uploaded-image inference now runs in the C# API with a bounded pool of OpenVINO requests; continuous RTSP frame extraction remains separate work.

A local-first multi-camera platform with a modern Next.js operations console, an ASP.NET Core control and OpenVINO inference API, PostgreSQL, Alembic, and Azurite Blob Storage.

The previous Axis camera code and the existing training assets were used as source material. The unsafe shell-built RTSP/OpenSSL flow and redundant Node streaming server were replaced; original model checkpoints and datasets remain under [`project/`](project/).

## Architecture

- **Next.js:** responsive multi-camera wall, authentication, camera setup, HLS playback, and server-side API proxies.
- **ASP.NET Core:** users, hashed bearer sessions, camera metadata, AES-GCM camera credentials, and one managed FFmpeg process per active RTSP-to-HLS stream.
- **C# inference:** `POST /api/inference/detect` uses native OpenCV/OpenVINO with pooled buffers and bounded admission. See [runtime configuration and validation](api/Inference/README.md).
- **PostgreSQL:** users, sessions, cameras, stream history, and inference event schema, versioned by Alembic.
- **Azurite:** local Azure Blob-compatible storage in place of MinIO.

## Quick start

```bash
cp .env.dev.example .env
./scripts/optimize-model.sh
./scripts/up-dev.sh
```

Open `http://localhost:3001`, the ASP.NET API at `http://localhost:5155/swagger`, with inference at `/api/inference/detect`.

The development stack starts PostgreSQL, Azurite, the Alembic migration runner, the API, and Next.js. Start the specialized ML service separately with `docker compose --env-file .env -f compose.dev.yml --profile ml up --build ml-api` after providing `kaggle.json` if dataset downloads are needed.

When opening the development UI from another device, set `NEXT_ALLOWED_DEV_ORIGINS` in `.env` to the hostname or LAN IP used in the browser (for example, `192.168.1.252`). Next.js uses this allowlist for its development HMR WebSocket; production does not expose HMR.

## Model optimization

`./scripts/optimize-model.sh` is the complete post-training release workflow. It starts the Azurite and training services, creates and accuracy-checks FP32, FP16, and INT8 OpenVINO variants from `model_training/output/runs/<run-id>/checkpoints/best_model_fp32.pth`, runs PyTorch, FP32, FP16 and INT8 project and official CityPersons evaluations, uploads an immutable checksum-verified release to Azurite, and refreshes `model_training/output/active-models/` for the C# API.

Choose a meaningful immutable release ID when preparing a release:

```bash
./scripts/optimize-model.sh \
  --release-id v2026-09-10.release1 \
  --calibration-samples 300 \
  --validation-samples 500
```

The complete local release is stored at `model_training/output/releases/<release-id>/`. Per-precision evaluation reports and latency benchmarks are stored in the release’s `evaluation/` and `benchmarks/` subdirectories. The immutable remote release is stored in the `computer-vision-models` container below `person_detector_ssd/releases/<release-id>/`; every uploaded file is read back and SHA-256 verified. See [the model-training release documentation](model_training/README.md#automated-model-release) for the exact layout and all overrides.

The C# API mounts `model_training/output/active-models` read-only at `/models` and selects `person_detector_int8.xml` by default. Recreate the API service after a successful release so it loads the newly promoted model:

```bash
docker compose --env-file .env -f compose.dev.yml up -d --build --force-recreate api
```

Benchmark again on deployment hardware and validate the recorded accuracy report before promoting a model outside this environment. The large training/notebook image remains available through the optional `ml` Compose profile and may use `/dev/dri/renderD128`.

## Production model training

Run the complete, self-contained training job with:

```bash
TRAINING_EPOCHS=100 ./scripts/run-production-training.sh
```

It fully validates CityPersons in Azurite, runs `getCityPersons.sh` when a complete published dataset is unavailable, trains the shared-head, anchor-free detector with visibility supervision and crowd-repulsion losses using PyTorch 2.8, evaluates the selected checkpoint, exports FP32/FP16 and calibrates INT8 with an accuracy gate, verifies all six XML/BIN files before benchmarking, and evaluates all three exported precisions, and publishes only a successful checksum-verified release. The immutable model files are saved in the `computer-vision-models` container at `person_detector_ssd/releases/<release-id>/`; `person_detector_ssd/current.json` records the latest complete FP32, FP16, and INT8 XML/BIN pairs. The application currently loads the promoted local copy; it does not watch Azurite for model updates. Logs, checkpoints, TensorBoard events, model files and per-precision reports live under the host-visible `model_training/output/` directory (`MODEL_OUTPUT_DIR` overrides it). Existing old artifacts and volumes are retained. Validation AP50/AP50:95 and Recall@FPPI are measured every epoch by default during the 60-epoch occlusion-training run, and the launcher exposes TensorBoard at `http://127.0.0.1:6006`. See [the production training documentation](model_training/README.md#production-one-shot-training-container) for settings and recovery behavior.

## Storage and migrations

Azurite is the local Azure Blob Storage emulator. The `computer-vision-data` and `computer-vision-models` containers are created by the camera storage adapter. `model_training/getCityPersons.sh` publishes immutable CityPersons versions below `datasets/citypersons/<version>/`, preserving pinned official source annotations and rich canonical JSON while generating standard binary-person YOLO labels. It validates every remote artifact and checksum before writing `datasets/citypersons/current.json` last. Visual previews are written to `model_training/validation_preview/<version>/contact_sheet.jpg`. Set `CITYPERSONS_RESET_CONTAINER=true` only for an intentional clean rebuild, `CITYPERSONS_DATASET_VERSION` to choose a version, and `CITYPERSONS_PREVIEW_COUNT` to change the preview count. Failed runs preserve their temporary work directory; resume by setting `CITYPERSONS_WORK_DIR` to the reported path.

PostgreSQL schema changes are managed by Alembic in [`migrations/`](migrations/).

## Production

Generate or provide the OpenVINO model artifacts, copy `.env.prod.example` to `.env`, replace every placeholder secret, and run:

```bash
docker compose --env-file .env -f compose.prod.yml up -d --build
```

Production images are built separately for the API, frontend, and the Alembic migration runner. The GPU/ML image remains opt-in and is intentionally excluded from the default CI image matrix.

## Workflow commands

```bash
./scripts/check-all.sh
./scripts/check-backend.sh
./scripts/check-python.sh
./scripts/check-migrations.sh
./scripts/check-images.sh
./scripts/optimize-model.sh
```

Never commit `.env`, `kaggle.json`, credentials, raw datasets, generated test output, or model artifacts that are not intentionally versioned. Generate a production camera key with `openssl rand -base64 32`; changing it later requires re-entering stored camera credentials. Kaggle credentials are mounted at runtime only.

## Contributing and security

See [`CONTRIBUTING.md`](CONTRIBUTING.md), [`SECURITY.md`](SECURITY.md), and [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md). See [`CHANGELOG.md`](CHANGELOG.md) for release notes.
