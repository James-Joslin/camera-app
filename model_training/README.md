# Person detector training and release workflow

This directory owns the CityPersons binary-person detector from immutable dataset ingestion through FP32 training, metric evaluation, and accuracy-controlled OpenVINO INT8 release. The exported model has one quality-aware person logit per anchor; canonical source labels remain available for sampling and metrics but are not exported as extra heads.

## What is implemented

- Canonical manifest loading with strict schema, checksum, class, and dataset-version validation.
- Aspect-preserving letterbox preprocessing shared by training, PyTorch inference, OpenVINO inference, and INT8 calibration.
- Pedestrian-shaped anchors, ATSS assignment, ignore-region neutralization, GIoU regression, and a one-logit Quality Focal Loss head whose target is aligned IoU.
- Training-only weighted oversampling for small, heavily occluded, rider, sitting, other-person, and unusual-pose records. Validation remains in natural manifest order.
- Project binary-person AP50, AP50:95, Recall@FPPI, and size/source-label/visibility slices.
- Checksum-pinned official CityPersons evaluation for Reasonable, Reasonable-small, heavy-occlusion, and All miss rates.
- Deterministic manifest-driven INT8 calibration with NNCF accuracy control, identical validation records across FP32/FP16/INT8, and core plus end-to-end latency reports.

## Architecture

The public executables stay compatible while replaceable behavior is separated behind small interfaces.

| Module | Responsibility |
| --- | --- |
| `person_detection/contracts.py` | Abstract inference, evaluation, and optimization boundaries |
| `person_detection/assignment.py` | `AnchorAssigner` contract and `ATSSAnchorAssigner` |
| `person_detection/sampling.py` | Record-selection and hard-case weighting policies |
| `canonical_dataset.py` | Immutable manifest dataset, shared preprocessing, and safe augmentation |
| `train_tune_detector.py` | Model/loss definitions and `DetectorTrainingPipeline` |
| `test_inference.py` | PyTorch/OpenVINO backends and `DetectionEvaluationWorkflow` |
| `citypersons_evaluation.py` | Official payload adapter and `PinnedCityPersonsEvaluator` |
| `optimize_model.py` | `ManifestDrivenOpenVINOOptimizer` release pipeline |

Abstract classes are used only where implementations are genuinely interchangeable: anchor assignment, manifest selection, inference backends, evaluation backends, and optimization pipelines.

## Start the environment

Run commands from the `camera-software` repository root:

```bash
docker compose -f docker-compose.yml -f compose.training.yml up -d azurite training
```

The training image mounts `model_training/` at `/workspace`. It contains the pinned PyTorch, torchvision, OpenVINO, NNCF, Albumentations, Azure, and Kaggle dependencies.

## Build and publish the canonical dataset

Place a valid `kaggle.json` at the repository root, then run:

```bash
docker compose -f docker-compose.yml -f compose.training.yml exec training ./getCityPersons.sh
```

The script downloads the source images and checksum-pinned official annotations, builds a version such as `datasets/citypersons/vYYYY-MM-DD`, uploads it to Azurite, validates every manifest relation and checksum, renders a preview, and publishes `datasets/citypersons/current.json` last.

Useful controlled overrides are:

```bash
docker compose -f docker-compose.yml -f compose.training.yml exec \
  -e CITYPERSONS_DATASET_VERSION=v2026-09-08.release1 \
  -e CITYPERSONS_WORK_DIR=/workspace/.citypersons-work \
  training ./getCityPersons.sh
```

Keep a work directory when an upload must be resumed. Set `CITYPERSONS_RESUME_UPLOAD=true` on the resume invocation; immutable-prefix protection otherwise rejects accidental replacement.

## Train FP32

```bash
docker compose -f docker-compose.yml -f compose.training.yml exec \
  -e ENABLE_QUANTIZATION=false training python train_tune_detector.py
```

`DetectorTrainingPipeline` reads the following environment variables:

- `AZURITE_BLOB_ENDPOINT`, `AZURITE_ACCOUNT_NAME`, `AZURITE_ACCOUNT_KEY`, and `AZURITE_CONNECTION_STRING`
- `AZURITE_DATA_CONTAINER` and `AZURITE_MODEL_CONTAINER`
- `USE_AZURITE` (`true` by default) and `DATA_ROOT` for local fallback
- `ENABLE_QUANTIZATION`, which must remain `false`

Model and optimizer hyperparameters live in `TrainingConfig`. Checkpoints contain `modelFormatVersion`, dataset version/checksum/schema provenance, sampling policy, ATSS settings, and head semantics. A compatible checkpoint is reused only when its immutable dataset identity also matches.

The main artifact is `best_model_fp32.pth`; the training pipeline also exports `person_detector_fp32.xml` and `.bin` when OpenVINO is available.

## Evaluate project and official metrics

First materialize the checksum-pinned evaluator files in a persistent workspace directory:

```bash
docker compose -f docker-compose.yml -f compose.training.yml exec training \
  python download_citypersons_annotations.py --output /workspace/.citypersons-official
```

Official CityPersons reporting requires the complete validation manifest, so use a maximum above the dataset size:

```bash
docker compose -f docker-compose.yml -f compose.training.yml exec training \
  python test_inference.py \
    --model-path best_model_fp32.pth \
    --model-type pytorch \
    --input-size 480 \
    --coco-map \
    --map-threshold 0.01 \
    --nms-threshold 0.5 \
    --recall-fppi 0.1 \
    --max-images 100000 \
    --official-evaluator-dir /workspace/.citypersons-official/evaluation/eval_script \
    --output-dir test_output
```

For a quick project-metric smoke test, omit `--official-evaluator-dir` and set a smaller `--max-images`. For an OpenVINO model, pass its `.xml` path with `--model-type openvino`.

Outputs include visualizations, `map_results.txt`, `evaluation_metrics.json`, official ground-truth/detection JSON, and the official evaluator text report. `evaluation_metrics.json` records dataset provenance, project metrics and slices, and official evaluator commit/checksums.

## Export, calibrate, validate, and benchmark INT8

```bash
docker compose -f docker-compose.yml -f compose.training.yml exec training \
  python optimize_model.py \
    --checkpoint best_model_fp32.pth \
    --output-dir optimized \
    --input-size 480 \
    --calibration-samples 300 \
    --validation-samples 500 \
    --selection-seed 1337 \
    --max-accuracy-drop 0.01 \
    --score-threshold 0.01 \
    --nms-threshold 0.5 \
    --benchmark-iterations 100 \
    --threads 1
```

Calibration comes only from clean train-manifest records and is greedily stratified across status, size, city, source label, posture, and visibility. Validation uses a deterministic natural-order subset. All selected image and annotation identities are written to `calibration_manifest.json` or `optimization_report.json`.

The optimizer creates FP32, FP16, and INT8 OpenVINO IR files, evaluates them on the same records, measures core and end-to-end p50/p95 latency, and rejects INT8 when the absolute AP50:95 loss exceeds `--max-accuracy-drop`. Dataset provenance is strict by default; `--allow-unverified-checkpoint` is an explicit diagnostic escape hatch and should not be used for release artifacts.

Do not set `ENABLE_QUANTIZATION=true` during training. The retired in-training QAT route intentionally fails fast; `optimize_model.py` is the sole release INT8 workflow.

## Tests

Run the complete model-training suite inside the dependency-pinned container:

```bash
docker compose -f docker-compose.yml -f compose.training.yml exec training \
  python -m unittest discover -v -p 'test_*.py'
```

The roadmap regressions cover sampling and slices, ATSS and ignore regions, quality targets and GIoU, legacy-head migration, official-evaluator provenance, BF16 loss/backpropagation, and the tiny-box CoarseDropout warning case.

## Operational notes

- Raw official boxes may cross an image boundary. They remain unchanged in canonical storage and official-evaluator payloads, then are clipped exactly once at the training/project-metric transform boundary.
- `BoxPreservingCoarseDropout` treats synthetic holes as appearance-only occlusion. It does not shrink or delete detection targets, avoiding Albumentations' zero-pixel-area visibility division.
- Quality targets are cast to the classification-logit dtype before indexed assignment, so CUDA BF16 autocast does not fail when IoU math remains FP32.
- Validation is never oversampled. Slice metrics neutralize out-of-slice people instead of counting valid people as background false positives.
- Legacy two-logit checkpoints are migrated to a one-logit head as person-minus-background log odds. New checkpoints use model format version 2.
