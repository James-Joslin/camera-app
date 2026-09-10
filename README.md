# Sentinel Camera Software

A local-first multi-camera platform with a modern Next.js operations console, an ASP.NET Core control API, a dedicated FastAPI/OpenVINO inference plane, PostgreSQL, Alembic, and Azurite Blob Storage.

The previous Axis camera code and the existing training assets were used as source material. The unsafe shell-built RTSP/OpenSSL flow and redundant Node streaming server were replaced; original model checkpoints and datasets remain under [`project/`](project/).

## Architecture

- **Next.js:** responsive multi-camera wall, authentication, camera setup, HLS playback, and server-side API proxies.
- **ASP.NET Core:** users, hashed bearer sessions, camera metadata, AES-GCM camera credentials, and one managed FFmpeg process per active RTSP-to-HLS stream.
- **FastAPI:** lazy-loaded OpenVINO inference over uploaded frames; the web proxy exposes `POST /api/inference/detect`.
- **PostgreSQL:** users, sessions, cameras, stream history, and inference event schema, versioned by Alembic.
- **Azurite:** local Azure Blob-compatible storage in place of MinIO.

## Quick start

```bash
cp .env.dev.example .env
./scripts/optimize-model.sh
./scripts/up-dev.sh
```

Open `http://localhost:3001`, the ASP.NET API at `http://localhost:5155/swagger`, and FastAPI at `http://localhost:8002/docs`.

The development stack starts PostgreSQL, Azurite, the Alembic migration runner, the APIs, and Next.js. Start the specialized ML service separately with `docker compose --env-file .env -f compose.dev.yml --profile ml up --build ml-api` after providing `kaggle.json` if dataset downloads are needed.

## Model optimization

`./scripts/optimize-model.sh` is the complete post-training release workflow. It starts the Azurite and training services, creates and accuracy-checks FP32, FP16, and INT8 OpenVINO variants from `model_training/best_model_fp32.pth`, runs the full FP32 project and official CityPersons evaluation, uploads an immutable checksum-verified release to Azurite, and refreshes `model_training/optimized/` for the FastAPI service.

Choose a meaningful immutable release ID when preparing a release:

```bash
./scripts/optimize-model.sh \
  --release-id v2026-09-10.release1 \
  --calibration-samples 300 \
  --validation-samples 500
```

The complete local release is stored at `model_training/releases/<release-id>/`. Evaluation previews remain at `model_training/release_evaluations/<release-id>/fp32/`. The immutable remote release is stored in the `computer-vision-models` container below `person_detector_ssd/releases/<release-id>/`; every uploaded file is read back and SHA-256 verified. See [the model-training release documentation](model_training/README.md#automated-model-release) for the exact layout and all overrides.

The lightweight FastAPI service mounts `model_training/optimized` read-only at `/models` and selects `person_detector_int8.xml` by default. Recreate the FastAPI service after a successful release so it loads the newly promoted model:

```bash
docker compose --env-file .env -f compose.dev.yml up -d --build --force-recreate fastapi
```

Benchmark again on deployment hardware and validate the recorded accuracy report before promoting a model outside this environment. The large training/notebook image remains available through the optional `ml` Compose profile and may use `/dev/dri/renderD128`.

## Storage and migrations

Azurite is the local Azure Blob Storage emulator. The `computer-vision-data` and `computer-vision-models` containers are created by the camera storage adapter. `model_training/getCityPersons.sh` publishes immutable CityPersons versions below `datasets/citypersons/<version>/`, preserving pinned official source annotations and rich canonical JSON while generating standard binary-person YOLO labels. It validates every remote artifact and checksum before writing `datasets/citypersons/current.json` last. Visual previews are written to `model_training/validation_preview/<version>/contact_sheet.jpg`. Set `CITYPERSONS_RESET_CONTAINER=true` only for an intentional clean rebuild, `CITYPERSONS_DATASET_VERSION` to choose a version, and `CITYPERSONS_PREVIEW_COUNT` to change the preview count. Failed runs preserve their temporary work directory; resume by setting `CITYPERSONS_WORK_DIR` to the reported path.

PostgreSQL schema changes are managed by Alembic in [`migrations/`](migrations/).

## Production

Generate or provide the OpenVINO model artifacts, copy `.env.prod.example` to `.env`, replace every placeholder secret, and run:

```bash
docker compose --env-file .env -f compose.prod.yml up -d --build
```

Production images are built separately for the API, FastAPI, frontend, and the Alembic migration runner. The GPU/ML image remains opt-in and is intentionally excluded from the default CI image matrix.

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
