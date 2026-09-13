# Pretrained YOLOv8n and YOLO26n benchmarks

Run from the repository root:

```bash
VALIDATION_SAMPLES=500 CALIBRATION_SAMPLES=300 BENCHMARK_ITERATIONS=100 \
OPENVINO_THREADS=1 MAX_ACCURACY_DROP=0.01 YOLO_RUN_ID=yolov8n-baseline \
./scripts/run-yolo-benchmark.sh
```

To benchmark YOLO26n with the same dataset and settings:

```bash
YOLO_MODEL=yolo26n YOLO_RUN_ID=yolo26n-baseline \
./scripts/run-yolo-benchmark.sh
```

`YOLO_MODEL` accepts `yolov8n` (default) or `yolo26n`. Weights and exported
artifacts are named for the selected model; both use the same cached dataset.

Or run `./run-benchmark.sh` from this directory. The job runs in the foreground
and returns a failing exit status on errors. Choose a new `YOLO_RUN_ID` each run;
existing run directories are never overwritten. The default combines the model name and a UTC timestamp.

The launcher starts Azurite, builds the pinned CPU container, verifies the
canonical CityPersons manifest and blob checksums, and reuses the existing
`getCityPersons.sh` download/build/upload workflow if validation fails. That
bootstrap needs Kaggle credentials (`KAGGLE_JSON_PATH`, default `./kaggle.json`)
and publishes a canonical dataset version just as the production training job
does. Existing valid data needs no download. The benchmark downloads and caches
the selected official COCO weights (`yolov8n.pt` or `yolo26n.pt`), exports OpenVINO FP32, verifies numerical
agreement with PyTorch, and evaluates the validation split. It also calibrates
INT8 on **training images only**, then evaluates INT8 on the same validation
selection. Set `YOLO_INT8=false` for just the pretrained FP32 benchmark.

There is **no fine-tuning**, optimizer, training loop, or detector release
publication. INT8 calibration is post-training quantization, not fine-tuning.
Both models retain their pretrained 80-class inference head. Only person detections
enter the project's evaluation. YOLO26 uses its native one-to-one NMS-free head;
the unused one-to-many training branch is removed during export. No parameters
are retrained.

## Input and comparison contract

- Public input: RGB float32 NCHW `[1,3,360,640]`, values divided by 255.
- The same aspect-preserving resize and black 640×360 letterbox as the current
  detector; **no ImageNet normalization**, since YOLO expects RGB /255.
- YOLO's stride needs a multiple of 32: the exported graph adds 24 black rows
  at the bottom, giving an internal 640×384 input. It does not stretch the scene.
  This padding and its compute are included in core latency. This is a deliberate
  project-comparison transform, not Ultralytics' default 114-filled letterbox.
- Both models use score ≥0.01, boxes clipped to the public canvas and a maximum
  of 100 person detections for accuracy evaluation.
- YOLOv8n uses person pre-NMS top-1000 and IoU 0.5 NMS.
- YOLO26n keeps native NMS-free output `[1,300,6]` (xyxy, probability, class):
  top-300 across all COCO classes inside the graph, followed by person filtering
  and the shared 100-person cap. No extra NMS is applied. This is a comparison
  of each model's native inference path, not identical postprocessing.
  The graph's native top-k work is included in YOLO26 core latency.
- AP50, AP50:95, Recall@FPPI=0.1 and slices use the shared project evaluator,
  canonical pedestrian labels and ignore regions. AP uses all-point precision
  interpolation. These are directly comparable to matching project evaluation
  settings, **not** the published 80-class COCO mAP or official CityPersons MR.
- Default validation is all 500 images in canonical sorted order. Smaller
  `VALIDATION_SAMPLES` values are smoke tests, not the full benchmark. Exact
  sample identities and manifest checksums are included in the report.
- OpenVINO CPU, batch 1, one stream, one inference thread by default. FP32 uses
  an f32 precision hint. INT8 results report the measured absolute AP50:95 drop
  versus FP32 and whether `MAX_ACCURACY_DROP` was met. A failed accuracy gate is
  recorded in the report; it does not delete results or fail this benchmark job.
- Core latency measures inference on a ready tensor. End-to-end latency includes
  compressed-image decode, RGB conversion, preprocessing, inference and person postprocessing (NMS for YOLOv8n),
  excluding storage/network access. It rotates through up to 16 cached images,
  with 8 warmups; `BENCHMARK_SCORE_THRESHOLD` defaults to 0.5 for timing only.

Run latency comparisons while other training/compute jobs are idle and use the
same hardware, threads, precision and postprocessing settings for both models.
The launcher leaves the existing training job alone. Halved training time does
not by itself establish an inference speedup.

## Model comparison — 2026-09-13

Benchmarked using the project's evaluator on all **500 validation images**, with **300 training images** for INT8 calibration. OpenVINO CPU timing used batch 1, one stream, one inference thread, 8 warmups and 100 measured iterations. Public canvas: **640×360**; YOLO additionally pads internally to **640×384**.

The trained person detector (`clean_ltrb`, C+E) uses **best_model_fp32.pth from epoch 57** (stored index 56) of the **60-epoch** run `v2026-09-12T220016Z`, selected by minimum validation loss (0.8901988). Its measurements were taken on an Intel Core i3-13100. The YOLO rows retain the pretrained model measurements recorded on this date; they were not rerun with the detector evaluation.

| Model | Precision | AP50 | AP50:95 | Recall @ FPPI 0.1 | Core mean ms | E2E mean ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Trained person detector (C+E) | FP32 | 0.3567 | 0.1403 | 0.1638 | 14.72 | 59.22 |
| Trained person detector (C+E) | FP16 | 0.3567 | 0.1399 | 0.1640 | 15.04 | 58.82 |
| Trained person detector (C+E) | INT8 | 0.3509 | 0.1381 | 0.1579 | 9.04 | 53.57 |
| YOLOv8n | FP32 | 0.4046 | 0.1731 | 0.2115 | 52.86 | 100.06 |
| YOLOv8n | INT8 | 0.3849 | 0.1524 | 0.2039 | 25.31 | 65.07 |
| YOLO26n | FP32 | 0.3820 | 0.1701 | 0.1892 | 36.51 | 91.25 |
| YOLO26n | INT8 | 0.3802 | 0.1653 | 0.1798 | 21.21 | 63.36 |

Core latency measures inference on a prepared tensor. E2E includes image decode, preprocessing, inference and person postprocessing, excluding storage/network access. Accuracy uses score ≥0.01; timing uses score ≥0.5. Models retain their respective preprocessing and native postprocessing paths described above. FP16 here means **compressed FP16 weights with FP32 CPU execution**, not native FP16 compute. YOLO FP16 was not measured.

### INT8 accuracy gates

Maximum permitted absolute AP50:95 drop: **0.01**.

| Model | AP50:95 drop | Gate |
| --- | ---: | --- |
| Trained person detector (C+E) | 0.002221 | PASSED |
| YOLOv8n | 0.02064 | FAILED |
| YOLO26n | 0.0048 | PASSED |

The trained detector's INT8 core inference is **1.63× faster** than its FP32 export. Its AP50:95 is lower than both pretrained YOLO models in this comparison.

### Trained detector latency details

| Precision | Core p95 ms | E2E p95 ms | Core FPS | E2E FPS | XML + weights MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP32 | 15.56 | 62.00 | 67.93 | 16.89 | 7.45 |
| FP16 | 16.10 | 61.13 | 66.49 | 17.00 | 4.00 |
| INT8 | 9.29 | 56.69 | 110.64 | 18.67 | 3.75 |

Independent PyTorch FP32 checkpoint evaluation: AP50 **0.356940**, AP50:95 **0.140553**, Recall@FPPI=0.1 **0.163281**. The comparison table uses the exported OpenVINO models' measured accuracy.

Full detector results, official CityPersons miss rates and software versions: [dated summary](../benchmark_results/ce-60-best-20260913/summary.md). Per-slice accuracy, dataset provenance, latency stages and quality gates: [optimization report](../benchmark_results/ce-60-best-20260913/models/optimization_report.json).

### NMS

**YOLOv8n** uses the traditional one-to-many detection output followed by **Non-Maximum Suppression (NMS)**.

**YOLO26n** uses a dual-head design. Its default one-to-many path also uses NMS, while its one-to-one head supports **end-to-end NMS-free inference** with `nms=False`.


## Results

Results are host-visible under `model_training/output/benchmarks/yolo/runs/<run-id>/`, including `benchmark.log`, validation outputs, exported YOLO models, `summary.md` and `report.json`. Weights and image caches live alongside the runs under `output/benchmarks/yolo/`. `MODEL_OUTPUT_DIR` overrides the shared output root; historical Docker volumes and prior results are retained.

Before launching, all trained-detector FP32, FP16 and INT8 `.xml` and `.bin` files must exist and be nonempty in `output/active-models/`. The container also verifies that OpenVINO can read each pair. Missing/invalid files stop benchmarking before dataset downloads. Set `TRAINED_MODELS_DIR` to the host path of a particular release's `models/` directory to benchmark against that artifact set. This preflight does not rerun the trained detector or change historical comparison rows.

Dependencies are pinned to Ultralytics 8.4.0 (supports both models) and the project's OpenVINO/NNCF/
Torch versions. The historical YOLOv8n run above used Ultralytics 8.3.200.
The report includes versions, weights SHA-256, dataset provenance,
input contract, export parity and timing scope. Reports have `status: complete`
only after all requested stages finish.

Sources: [YOLO26 model documentation](https://docs.ultralytics.com/models/yolo26/),
[pinned YOLO26 detection head](https://github.com/ultralytics/ultralytics/blob/v8.4.0/ultralytics/nn/modules/head.py),
[YOLOv8 model documentation](https://docs.ultralytics.com/models/yolov8/),
[pinned detection-head implementation](https://github.com/ultralytics/ultralytics/blob/v8.3.200/ultralytics/nn/modules/head.py),
[official pretrained asset](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt).

Tests (after building the image):

```bash
docker run --rm --network none --entrypoint python camera-software-yolo:benchmark \
  -m unittest yolo_benchmark.test_benchmark yolo_benchmark.test_workflow
```
