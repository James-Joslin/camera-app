# Changelog

All notable changes to this project are documented here.

## [Unreleased]

### Added

- Modern responsive camera wall with HLS playback, expandable feeds, account flows, camera setup, and inference status.
- Structured ASP.NET repositories and endpoint groups for users, sessions, cameras, and stream lifecycle.
- AES-GCM camera credential encryption, PBKDF2 password hashing, hashed bearer sessions, and argument-safe FFmpeg process management.
- FastAPI OpenVINO detector endpoint plus reproducible FP16/INT8 conversion, calibration, and benchmark tooling.
- PostgreSQL camera-platform schema managed by Alembic.
- Generated OpenVINO FP16 and INT8 artifacts with a measured 1.77x INT8 CPU latency speedup.
- Next.js frontend, ASP.NET Core API, and FastAPI service scaffolds.
- PostgreSQL/Alembic migration runner.
- Development and production Compose stacks with health checks.
- Azurite Blob Storage service, Azure Blob adapter, and recursive upload helper.
- Sample environments, workflow scripts, contribution and security docs, and GitHub automation.

### Changed

- Replaced the previous shell/OpenSSL/GStreamer camera flow and redundant Node streaming tier with the managed C# stream service.
- Upgraded the frontend to Next.js 16.3.4 and removed external font loading; npm audit reports zero known vulnerabilities.
- Mounted optimized models read-only into the non-root FastAPI production container.
- Removed MinIO client/tooling and hard-coded database/object-storage credentials from the ML image.
- Kept the existing ML/training assets behind an opt-in Compose `ml` profile.

