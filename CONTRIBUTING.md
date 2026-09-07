# Contributing

1. Copy `.env.dev.example` to `.env`.
2. Run `./scripts/check-all.sh` before opening a pull request.
3. Run `./scripts/optimize-model.sh` and review `project/optimized/benchmark.json` when detector deployment changes; validate accuracy as well as latency.
4. Keep stream and inference routes behind authentication when adding externally reachable deployment infrastructure.
5. Keep PostgreSQL schema changes in a new Alembic revision.
6. Keep secrets, `kaggle.json`, raw data, generated output, and local model files out of commits.
7. Describe any camera/ML hardware assumptions, migration impact, and test commands in the pull request.

Use focused commits and update [`CHANGELOG.md`](CHANGELOG.md) for user-visible changes.

