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

`./scripts/optimize-model.sh` converts `project/best_model_fp32.pth` to OpenVINO FP16 and INT8, calibrates INT8 from `project/test_output`, and writes artifacts plus `project/optimized/benchmark.json`. Override defaults by passing optimizer arguments, for example `./scripts/optimize-model.sh --calibration-samples 256 --benchmark-iterations 200`.

The checked run used 128 calibration images at 480 × 480. On this host, FP16 averaged **8.29 ms (120.6 FPS)** and INT8 averaged **4.69 ms (213.3 FPS)** over 100 runs: a **1.77× latency speedup**. These are single-request CPU numbers; benchmark again on deployment hardware and validate accuracy before promoting a model.

The lightweight FastAPI service mounts `project/optimized` read-only at `/models`. The large training/notebook image remains available through the optional `ml` Compose profile and may use `/dev/dri/renderD128`.

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
