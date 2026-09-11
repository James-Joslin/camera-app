# Person Detector Architecture Improvement Strategy

## Executive recommendation

The product foundations needed before changing the network are now implemented: a fixed 360×640 H×W landscape canvas, rectangular anchors and decoding, resumable training, explicit experimental/production release states, quality floors, and a stage-by-stage latency benchmark with bounded pre-NMS candidates.

The immediate priority is therefore to train the model-format-v3 rectangular detector long enough to establish a trustworthy experimental baseline. It does not need to reach epoch 100 before it can be evaluated or packaged as `experimental`: the run now resumes from the last completed epoch. It must not be labelled `production` until both the model and representative camera footage pass the new quality gates.

The remaining progression is:

1. Train and measure the current 360×640 anchor-based control without mixing in network changes.
2. Measure person width/height distributions, rectangular-anchor coverage, ATSS positives, and recall by size and occlusion on CityPersons and representative cameras.
3. Validate the implemented top-1,000 pre-NMS cap and the new production-threshold latency results on that trained artifact.
4. Decide the C# / FastAPI / OpenVINO preprocessing boundary from JPEG and raw-frame measurements; keep geometry and normalization owned by one canonical contract.
5. Replace dense FPN smoothing with a depthwise-separable variant, then test one lightweight weighted bidirectional fusion pass.
6. Simplify and share the detection tower before considering more attention.
7. Build an anchor-free branch only after the anchor-based rectangular control is measured.
8. Add crowd-aware loss/suppression and teacher distillation only when slice metrics show they are needed.

A transformer or NMS-free detector is not the next move. The dataset is small, the current quality-head model has not yet produced a valid baseline checkpoint, and the existing architecture has simpler, measurable inefficiencies.

## Implementation status

“Implemented” below means the code path and contracts exist and have passed unit/smoke checks. It does not mean a newly trained rectangular checkpoint has already demonstrated production accuracy.

| Area | Status | Current result |
| --- | --- | --- |
| 360×640 H×W landscape input | Implemented | Independent height/width flow through dataset loading, augmentation, calibration, export, evaluation, optimization, and FastAPI serving; non-landscape configurations fail fast. |
| Rectangular anchors and boxes | Implemented | Anchors use actual feature-map `(height, width)`, are constructed in canvas pixel space, and normalize x/width separately from y/height. The current output has 29,235 predictions. |
| Letterbox and inverse mapping | Implemented | Aspect ratio is preserved; boxes are transformed and mapped back per axis without square-canvas assumptions. |
| Interrupted-run recovery | Implemented | An atomic last checkpoint stores model, optimizer, scheduler, AMP scaler, epoch/completion, best loss, sampler state, and RNG state. Only a compatible completed run is skipped. |
| Release safety | Implemented | Releases are explicitly `experimental` or `production`; production requires minimum FP32 and INT8 metrics plus representative camera-domain acceptance. |
| Latency benchmark | Implemented | Accuracy remains at threshold 0.01; latency defaults to 0.5 and reports JPEG decode, color conversion, letterbox/normalization, inference, box decode/filter, and NMS separately, including a raw-BGR path. |
| NMS candidate control | Implemented | Score filtering occurs before decoding/NMS, then top 1,000 candidates are retained by default. Capped and uncapped validation are compared before release. |
| Pedestrian/anchor ratio evidence | Pending measurement | The physical ratios are preserved on the new canvas, but CityPersons/camera distributions, coverage, ATSS-positive counts, and recall slices still need a trained v3 artifact and analysis. |
| OpenVINO-owned preprocessing | Deferred | The current IR accepts normalized NCHW tensors. Moving layout/color/normalization into OpenVINO should follow measurement and an explicit C# payload contract. |
| Asynchronous multi-camera inference | Out of scope | C# frame extraction/routing, OpenVINO async requests/streams, tracking, and removal of the FastAPI global serialization lock are intentionally unchanged. |
| Neck/head/anchor-free changes | Deferred | Dense FPN smoothing, per-level towers, redundant normalization/activation, attention, and anchor multiplication remain the architecture experiments below. |

## Scope and constraints

This assessment assumes:

- Binary person detection from fixed cameras.
- CityPersons as the current supervised dataset: 2,975 training images and 500 validation images.
- A 360×640 H×W current input (equivalently 640×360 W×H), CPU inference through OpenVINO, and manifest-driven INT8 calibration.
- Small and occluded people, low false positives, and official CityPersons miss rate matter more than generic COCO AP.
- Architecture changes must remain exportable and quantizable with ordinary convolutional operators.

CityPersons was created specifically to expose scale, occlusion, and cross-dataset generalization problems in pedestrian detection.[^1] Recommendations therefore prioritize spatial efficiency, high-resolution features, crowd behavior, and a trustworthy-quality score over adding generic model capacity.

## Current architecture

The implementation in `person_detection/training/pipeline.py` is a 2.743-million-parameter SSD-style detector:

| Component | Parameters | Share |
| --- | ---: | ---: |
| MobileNetV3-Small feature extractor | 927,008 | 33.8% |
| Two extra pyramid levels | 460,480 | 16.8% |
| Five-level FPN | 870,912 | 31.8% |
| Five detection heads | 484,510 | 17.7% |
| Total | 2,742,910 | 100% |

The model taps MobileNetV3-Small at three stages, adds stride-64 and stride-128 features, applies a 128-channel top-down FPN, and predicts on five levels. At 360×640, the actual rectangular feature maps contain 4,835 spatial locations and emit 29,235 anchors:

| Level | Feature map | Locations | Anchors/location | Predictions |
| --- | ---: | ---: | ---: | ---: |
| stride 8 | 45×80 | 3,600 | 6 | 21,600 |
| stride 16 | 23×40 | 920 | 6 | 5,520 |
| stride 32 | 12×20 | 240 | 6 | 1,440 |
| stride 64 | 6×10 | 60 | 9 | 540 |
| stride 128 | 3×5 | 15 | 9 | 135 |
| Total | — | 4,835 | — | 29,235 |

The current output shapes are `[B, 29235, 1]` for the joint person-quality logit and `[B, 29235, 4]` for box offsets.

### Strengths worth retaining

- A stride-8 prediction level for small pedestrians.
- Multi-scale feature fusion. FPN established the value of semantically strong high-resolution features for scale-varying detection.[^2]
- ATSS assignment. Its central finding is that sample selection matters more than the nominal anchor/anchor-free distinction, and it adds no inference overhead.[^3]
- A single joint person/localization-quality logit. This follows the consistency argument behind Generalized Focal Loss: the ranking score should express both classification and localization quality.[^4]
- GIoU regression, ignore-region neutralization, hard-case sampling, and immutable metric provenance.
- A conventional convolutional operator set that is suitable for OpenVINO INT8.

### Weaknesses visible in the code

1. **The original square canvas wasted spatial budget (resolved).** CityPersons frames are 2048×1024. The former 480×480 canvas produced 480×240 content with 50% padding. The implemented 360×640 canvas produces 640×320 content with 20 pixels of padding above and below, while retaining the same 230,400-pixel tensor budget.

2. **The neck is disproportionately expensive.** Five dense 3×3, 128-to-128 FPN smoothing convolutions contain 737,280 kernel weights and require about 709 million multiply-accumulates across the five maps. A depthwise 3×3 plus pointwise 1×1 replacement would use about 87,680 kernel weights and 84 million multiply-accumulates for the same maps, before normalization: roughly an 8.4× reduction for this part of the neck. Actual OpenVINO latency must still be measured because operator efficiency is hardware-specific.

3. **The head stacks normalization and nonlinearities redundantly.** Each classification and regression subnet calls a separable block that already contains BatchNorm and LeakyReLU, then immediately applies GroupNorm and SiLU. This creates two normalizations and two activations around one feature transform. It is difficult to justify statistically, complicates quantization, and prevents a clean Conv–BatchNorm folding path.

4. **Every level has an independent head.** Five separate towers consume parameters and allow small CityPersons training splits to learn inconsistent level-specific features. The final output convolutions must remain level-specific while anchor counts differ, but most tower weights can be shared.

5. **The attention path is branch-heavy.** Every pyramid level performs average pooling, max pooling, two channel gates, channel reductions, a 7×7 spatial convolution, and sigmoid gates. These operations may help accuracy, but they are not free merely because parameter counts are small. MobileOne’s deployment study is a useful warning that FLOPs and parameter counts can correlate poorly with real latency, and that branching and activation choices matter.[^5]

6. **The FPN is top-down and unweighted only.** All lateral inputs are summed equally, and information does not travel back from high-resolution maps toward lower-resolution maps. EfficientDet found that learnable weighted, bidirectional fusion can improve the accuracy/efficiency frontier.[^6]

7. **Anchor multiplication is no longer clearly earning its cost.** The ATSS paper itself reports that tiling multiple anchors per location may not be necessary once sample assignment is strong.[^3] Here it multiplies candidates by six or nine, increasing final convolution width, target assignment, decoding, thresholding, and NMS.

8. **Crowd structure is addressed mainly by ignore masks.** Fixed hard NMS at IoU 0.5 can suppress two nearby pedestrians, and ordinary GIoU does not explicitly prevent a box from drifting toward an adjacent person.

9. **Visible-box supervision is unused by the model.** Visible boxes affect augmentation and metadata slices, but there is no training signal that asks the feature extractor to represent visible extent or occlusion. This is an opportunity only after the visible-box data contract is corrected and exhaustively validated.

## Highest-priority improvements

## 1. Rectangular, deployment-shaped input — implemented

The active contract is a fixed **360×640 H×W** landscape canvas across training and serving. Width is deliberately greater than height. For the current CityPersons source geometry, this retains the 640×320 image content and adds only 20 pixels of padding above and below.

The spatial-budget comparison that motivated the change remains useful:

| Canvas | Pixels | CityPersons content | Padding | Scale from 2048×1024 |
| --- | ---: | ---: | ---: | ---: |
| Historical 480×480 | 230,400 | 480×240 | 50.0% | 0.234× |
| 640×320 | 204,800 | 640×320 | 0% | 0.3125× |
| Implemented 640×360 | 230,400 | 640×320 | 11.1% | 0.3125× |
| 768×384 | 294,912 | 768×384 | 0% | 0.375× |

At 640×360, a 50-pixel source pedestrian becomes about 15.6 pixels tall rather than 11.7 pixels, a 33% linear increase with the same tensor pixel count as 480×480. This is still an analytical projection, not an accuracy claim; a newly trained checkpoint must establish the gain.

The implementation now:

- Replaces scalar `input_size` with `input_height` and `input_width` in active training, preprocessing, augmentation, calibration, export, decoding, evaluation, optimization, and serving paths.
- Preserves aspect ratio with deterministic letterboxing and carries scale/padding metadata for source-image inverse mapping.
- Derives anchors from the model's real per-level `(height, width)` feature tensors instead of assuming square maps.
- Creates anchor dimensions in canvas pixel space using the square root of canvas area as the scale reference, then normalizes x/width by 640 and y/height by 360. Changing canvas shape therefore does not silently change physical width/height ratios.
- Decodes and clips each axis using its corresponding canvas dimension.
- Exports and validates a static OpenVINO input of `[1, 3, 360, 640]`; FastAPI reads both dimensions from the compiled model and rejects a non-landscape input.
- Records the rectangular input and anchor specification in model-format-v3 checkpoints, optimization reports, and release metadata.

The model topology itself is otherwise unchanged: stride 8 is retained, and no stride-4 head has been added. The remaining work for this item is empirical rather than structural: train v3, compare optional 320×640 only if useful, verify round-trip boxes on representative frames, and measure small/heavy-occlusion metrics.

## 2. Lightweight bidirectional neck

Replace the current FPN output convolutions with depthwise-separable Conv–BatchNorm–SiLU blocks. Then compare:

- **Variant A:** the same one-pass top-down FPN with separable smoothing.
- **Variant B:** one weighted top-down plus bottom-up BiFPN pass with separable convolutions.
- **Variant C:** two BiFPN passes only if one pass produces a clear accuracy gain.

BiFPN’s evidence supports bidirectional, weighted cross-scale fusion,[^6] but the specific number of repeats must be chosen on this CPU and dataset. One 96- or 128-channel pass is a sensible starting point. Preserve the stride-8 map because small-person performance is a first-class metric.

Do not add a large PAN/BiFPN stack by default. The current neck already represents nearly one-third of model parameters, and the high-resolution convolution dominates arithmetic. The goal is better fusion while reducing, not merely relocating, compute.

## 3. Clean, shared detection tower

Use one explicit block definition, for example:

```text
depthwise 3×3 → BatchNorm → SiLU → pointwise 1×1 → BatchNorm → SiLU
```

Then build a shared two-block tower with separate classification and regression branches. Share each branch across levels and retain small level-specific output convolutions. If effective batch size becomes too small for stable BatchNorm, use GroupNorm instead—but never both consecutively around the same convolution.

Run attention as an ablation:

1. No explicit channel or spatial attention.
2. Channel-only lightweight attention.
3. Current channel-plus-spatial attention.

The simplest head should be the baseline. Retain attention only if it improves official miss rate or AP50:95 enough to justify measured INT8 latency and calibration impact. OpenVINO supports GroupNormalization, but supported does not mean optimal; its low-precision pipeline is explicitly designed to propagate and fuse quantized linear operations where possible.[^8]

## 4. Anchor-free pedestrian head

Create a separate experimental model rather than mutating the stable anchor-based checkpoint format. Predict once per location:

- One joint person/localization-quality logit.
- Four positive l/t/r/b distances from the location to box edges.
- Optionally one training-only visibility scalar.

At the current feature sizes, one prediction per location reduces candidates from 29,235 to 4,835—an 83.5% reduction. It also removes hand-designed scales and aspect ratios. FCOS established direct per-location l/t/r/b regression as a strong anchor-free design,[^9] while pedestrian-specific CSP showed that center-and-scale prediction can be both structurally simple and competitive on pedestrian benchmarks.[^10]

Two target-assignment options are credible:

- **ATSS-point:** retain ATSS’s statistical selection over locations. This minimizes conceptual change and is supported by ATSS experiments on anchor-free detectors.[^3]
- **Task-aligned assignment:** rank candidates with a combination of predicted classification and localization quality. TOOD found that explicit task alignment addresses classification/localization spatial mismatch and improved over ATSS and GFL in its COCO experiments.[^11]

Start with direct l/t/r/b regression and GIoU. Distributional box regression from GFL can improve localization modeling,[^4] but it expands the regression output and adds softmax/integral operations. Test it only after the direct-regression anchor-free baseline is stable and quantized.

The joint quality score should remain. A separate centerness branch may be redundant when the single score is already trained toward IoU; add it only through an ablation.

## 5. Crowd-aware localization and suppression

Before introducing an occlusion transformer or part-based network, test two lower-risk changes:

- Add Repulsion Loss beside GIoU only for images containing overlapping people. Repulsion Loss was designed on CityPersons to keep a prediction close to its assigned person and away from neighboring people, with reported gains in occlusion cases.[^12] It is training-only and adds no deployment operators.
- Compare hard NMS with Gaussian or linear Soft-NMS. Soft-NMS decays overlapping scores instead of deleting candidates and reported consistent gains without retraining.[^13]

If crowd errors remain dominant, Adaptive NMS is a later option; it learns a density score and adjusts suppression thresholds, with published CityPersons and CrowdHuman improvements.[^14] It adds another prediction and more deployment logic, so it should follow the Soft-NMS experiment rather than precede it.

## 6. Distillation before a larger deployed backbone

The current supervised set is small for comparing many high-capacity architectures. Train a stronger teacher—such as a 768×384 model, MobileNetV3-Large, or a wider BiFPN—and distill localization distributions and quality scores into the compact student.

Localization Distillation reported improvements to dense detectors without student inference cost and found localization knowledge especially valuable.[^15] Multi-scale aligned distillation specifically targets the loss suffered by low-resolution detectors.[^7] This matches the deployment problem better than permanently increasing student width.

Subject to dataset licensing and label-policy review, CrowdHuman pretraining is also highly relevant: it contains roughly 470,000 human instances, about 22.6 people per image, and full/visible/head annotations, and its authors reported cross-dataset gains on CityPersons.[^16] This is more likely to improve crowd and occlusion representation than adding an untrained attention module to 2,975 images.

## Backbone strategy

Do not replace the backbone first. Only 33.8% of current parameters are in MobileNetV3-Small, while the extra levels and FPN together account for 48.6%. Fix spatial utilization and the neck before attributing limitations to the backbone.

After those changes, benchmark three categories:

| Candidate | Role | Rationale | Risk |
| --- | --- | --- | --- |
| MobileNetV3-Small | Control/student | Already exported and quantized | Accuracy ceiling |
| MobileNetV3-Large | Teacher or accuracy variant | Available in the same torchvision family | Higher latency |
| MobileOne-S0/S1 | Student candidate | Deployment-oriented re-parameterized CNN with detection evidence[^5] | Conversion and pretrained-weight integration |
| RepViT-M0.9/M1.0 | Student candidate | Mobile CNN design with strong reported latency/accuracy[^17] | Results are not OpenVINO-CPU measurements |
| MobileNetV4-Conv-S | Research candidate | Newer universal inverted bottleneck family designed across accelerators[^18] | Toolchain/model availability and export maturity |

Latency claims from phones, GPUs, or EdgeTPUs must not be transferred to this CPU. Export each candidate to the same OpenVINO version, calibrate it from the same manifest records, and measure core plus end-to-end latency.

Avoid a transformer backbone at this stage. The expected dataset and deployment benefits do not justify the additional export, calibration, and memory uncertainty when simpler CNN inefficiencies remain.

## Visibility-aware auxiliary supervision

Once zero-area visible boxes have defined semantics, add an optional training-only auxiliary target on positive locations:

- Visible fraction in `[0,1]`, or
- A coarse visibility class: clear, partial, heavy.

The head should be removed from the exported inference graph unless its output is operationally needed. This lets visible-box labels shape shared features without adding production latency. An explicit part-confidence approach has previously improved single-stage pedestrian detectors on CityPersons, especially under heavy occlusion,[^19] but the current dataset is not yet clean enough to make this a first-phase change.

Do not predict visible-box coordinates initially. A scalar visibility objective is simpler, handles zero visible area naturally, and is less sensitive to noisy boundary annotations.

## Proposed next-generation architecture — not implemented

The block below is the target for later controlled experiments, not a description of the current model. Completed foundation work is labelled explicitly.

```text
RGB frame
  ↓
fixed 360×640 letterbox and normalization                  [implemented]
  ↓
MobileNetV3-Small control backbone                         [current]
  ├── stride 8
  ├── stride 16
  └── stride 32
  ↓
lightweight stride-64 and stride-128 blocks                [current]
  ↓
one 96/128-channel separable weighted BiFPN pass           [planned]
  ↓
shared decoupled separable towers                          [planned]
  ├── joint person × IoU score: 1 value/location
  ├── l/t/r/b box distances: 4 values/location
  └── visibility auxiliary: training only
  ↓
score floor → top-K → decode → Soft-NMS/hard-NMS experiment
              [top-K implemented]       [Soft-NMS planned]
```

This target keeps the completed rectangular spatial efficiency and the current strengths—stride-8 detail, quality-aware ranking, ignore regions, GIoU, and INT8-friendly convolution—while targeting the remaining dense neck convolutions, redundant normalization, duplicated towers, and anchor multiplication.

## Experiment roadmap

Do not combine model changes into one long run. The current CPU training time is about 20 minutes per epoch, and the new recovery checkpoint allows useful evidence to accumulate without restarting after an interruption.

| Order | Experiment | Status | Change from control / primary question |
| ---: | --- | --- | --- |
| 0 | B0 | Historical reference only | The 480×480 model-format-v2 report explains the previous behavior but is not the new baseline. |
| 1 | R2 | Code complete; training pending | Train the unchanged quality-head topology at 360×640. What are its real accuracy, slice recall, and latency? |
| 2 | V1 | Pending evidence | Measure full-body ratios, anchor coverage, ATSS positives, and recall by size/occlusion on CityPersons and camera footage. Can wide anchors be removed safely? |
| 3 | P1 | Tooling complete; artifact run pending | Compare top-1,000 with uncapped post-processing at threshold 0.01, then benchmark at 0.5. Does the cap preserve quality and reduce NMS cost? |
| 4 | O1 | Deferred until P1 | Benchmark JPEG and raw camera frames and choose whether C#, FastAPI, or OpenVINO owns decode, resize, layout conversion, and normalization. |
| 5 | R1 | Optional | Compare 320×640 H×W only if exact CityPersons aspect ratio could justify another long input-shape run. |
| 6 | N1 | Planned | Use separable FPN smoothing only. How much neck latency/size can be removed safely? |
| 7 | N2 | Planned | Add one separable weighted BiFPN pass. Does bidirectional fusion recover or improve accuracy? |
| 8 | H1 | Planned | Use a clean shared head without attention. Are double normalization and per-level towers unnecessary? |
| 9 | H2 | Optional ablation | Add lightweight attention back only if H1 loses important slice accuracy. |
| 10 | C1/C2 | Planned after error analysis | Test Repulsion Loss and then Soft-NMS when crowded-person errors justify them. |
| 11 | A1/A2 | Next-generation branch | Replace anchors with direct l/t/r/b regression, then consider task-aligned assignment. Can candidates fall 83.5% without recall loss? |
| 12 | D1/K1 | Later | Use teacher distillation, then compare backbone finalists on the actual OpenVINO CPU. |

The viable-product path is R2 → V1 → P1. It can produce an `experimental` artifact before all requested epochs finish, while preserving the ability to resume training. N1 and later network changes are not prerequisites for that artifact. A `production` release still requires the enforced model-quality and camera-domain evidence.

Use one seed for screening and at least three seeds for architecture finalists because 2,975 training images make small metric differences noisy. Preserve the same manifest, augmentation policy, schedule, evaluation threshold, calibration records, and release status across comparisons.

## Acceptance gates

### Automated artifact gates — implemented

The optimization/release pipeline now separates technical packaging from production readiness:

- Accuracy evaluation always uses the low `0.01` score threshold; the latency benchmark defaults to the production-like `0.5` threshold.
- INT8 must remain within `0.01` absolute AP50:95 of its own FP32 source by default.
- Top-1,000 and uncapped validation are compared at the evaluation threshold; the default maximum permitted quality drop is `0.0001`.
- `experimental` is the default release status. It still requires successful quantization and top-K gates, but it may honestly package an incomplete or below-target model.
- `production` requires both FP32 and INT8 to meet the configured minimums. Defaults are AP50 ≥ 0.25, AP50:95 ≥ 0.10, and recall at FPPI 0.10 ≥ 0.20.
- `production` also requires representative camera-domain metrics that meet the same configured minimums.
- The chosen release status, all gate evidence, input geometry, anchor specification, preprocessing contract, and artifact hashes are carried into the optimization report and publication metadata.

These are safety floors, not claims that the default numbers define an ideal product. They prevent a tiny INT8-vs-FP32 drop from approving a model whose FP32 baseline is itself unusably weak.

### Architecture comparison evidence

Every architecture variant should report:

- Project AP50 and AP50:95.
- Recall at FPPI 0.10.
- Small, medium, large, clear, partially occluded, and heavily occluded slices.
- Official Reasonable, Reasonable-small, heavy-occlusion, and All log-average miss rate.
- FP32/FP16/INT8 model bytes.
- OpenVINO core and stage-by-stage JPEG/raw-frame p50/p95 latency at fixed thread count.
- Candidates before score filtering, after filtering, entering NMS, and retained after NMS.
- INT8 accuracy drop against that variant's own FP32 result.
- Parameters, MACs, and peak preprocessing/inference memory.

Suggested screening rules for later architecture experiments:

- A simplification is successful if it reduces end-to-end p50 latency by at least 10% or model size by at least 15%, while worsening official Reasonable miss rate by no more than 0.3 percentage points and AP50:95 by no more than 0.5 points.
- An accuracy feature is successful if it improves heavy-occlusion or Reasonable-small miss rate by at least 1 percentage point without worsening end-to-end p95 latency by more than 10%.
- A backbone is acceptable only after successful OpenVINO FP16 and INT8 export and all current release gates.
- A result that appears only at one seed should be treated as provisional.

The architecture thresholds are engineering recommendations rather than published standards. Adjust them once explicit single-frame latency, aggregate camera FPS, and memory budgets exist.

## Baseline warning

The checked-in `optimized/optimization_report.json` is a historical square-canvas report. It identifies a model-format-v2 checkpoint, scalar 480 input, schema-v2 report, and a small INT8 degradation of about 0.0023 AP50:95. It predates rectangular inputs, the stage-by-stage benchmark, top-K validation, absolute quality floors, camera-domain acceptance, and release statuses. Its “accepted” result therefore does not establish that either the old FP32 model or the current architecture is production-viable.

The active code now requires model format v3 because input geometry and anchor semantics changed. A v2 480×480 checkpoint cannot safely resume into the 360×640 model: tensor geometry, anchor metadata, and optimizer trajectory are incompatible. Begin one v3 rectangular run. After that run completes its first epoch, future interruptions can resume model, optimizer, scheduler, scaler, sampling, RNG, best-loss, and epoch state from `last_training_checkpoint.pth`.

Until a v3 artifact has been trained and measured, the 29,235-prediction figures in this document describe the implemented topology, not achieved accuracy or latency.

## What not to prioritize

- **A large transformer or DETR conversion:** too much toolchain and data risk before simpler inefficiencies are removed.
- **More attention blocks:** the current architecture already has attention at every level without evidence that it earns latency.
- **A stride-4 prediction head immediately:** likely useful only if rectangular input and improved fusion still leave small-person recall deficient.
- **Distributional regression immediately:** credible for localization, but it increases output width; establish the anchor-free direct-regression baseline first.
- **NMS-free dual-assignment training immediately:** methods such as YOLOv10 target end-to-end latency,[^20] but add assignment and training complexity that is premature for this small binary dataset.
- **Changing backbone, neck, head, assignment, loss, and input shape together:** it prevents causal measurement and makes INT8 regressions difficult to diagnose.

## Conclusion

The highest-value foundation work is complete. The repository now has one landscape 360×640 geometry contract, physically stable rectangular anchors, genuine interruption recovery, honest experimental/production release states, bounded NMS input, and a benchmark capable of showing where time is spent.

For a viable product before the full training schedule finishes, start the v3 run, let the recovery checkpoint accumulate epochs, and evaluate/package selected checkpoints as `experimental`. Use the new ratio/coverage analysis, camera-domain metrics, and stage timings to decide whether more training or a targeted change is needed. Do not represent the artifact as `production` until its FP32, INT8, and camera results pass the production gates.

The first network experiment after that evidence should be separable FPN smoothing, followed by a clean shared head. The anchor-free branch can later reduce 29,235 anchor predictions to about 4,835 point predictions. OpenVINO preprocessing ownership and multi-camera asynchronous throughput remain separate deployment projects; they should not block the current model baseline or be mixed into its architecture comparison.

## Sources

[^1]: Zhang, Shanshan, Rodrigo Benenson, and Bernt Schiele. “[CityPersons: A Diverse Dataset for Pedestrian Detection](https://openaccess.thecvf.com/content_cvpr_2017/html/Zhang_CityPersons_A_Diverse_CVPR_2017_paper.html).” CVPR, 2017.
[^2]: Lin, Tsung-Yi, et al. “[Feature Pyramid Networks for Object Detection](https://openaccess.thecvf.com/content_cvpr_2017/html/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.html).” CVPR, 2017.
[^3]: Zhang, Shifeng, et al. “[Bridging the Gap Between Anchor-Based and Anchor-Free Detection via Adaptive Training Sample Selection](https://openaccess.thecvf.com/content_CVPR_2020/html/Zhang_Bridging_the_Gap_Between_Anchor-Based_and_Anchor-Free_Detection_via_Adaptive_CVPR_2020_paper.html).” CVPR, 2020.
[^4]: Li, Xiang, et al. “[Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection](https://papers.neurips.cc/paper_files/paper/2020/hash/f0bda020d2470f2e74990a07a607ebd9-Abstract.html).” NeurIPS, 2020.
[^5]: Vasu, Pavan Kumar Anasosalu, et al. “[MobileOne: An Improved One Millisecond Mobile Backbone](https://openaccess.thecvf.com/content/CVPR2023/html/Vasu_MobileOne_An_Improved_One_Millisecond_Mobile_Backbone_CVPR_2023_paper.html).” CVPR, 2023.
[^6]: Tan, Mingxing, Ruoming Pang, and Quoc V. Le. “[EfficientDet: Scalable and Efficient Object Detection](https://openaccess.thecvf.com/content_CVPR_2020/html/Tan_EfficientDet_Scalable_and_Efficient_Object_Detection_CVPR_2020_paper.html).” CVPR, 2020.
[^7]: Qi, Lu, et al. “[Multi-Scale Aligned Distillation for Low-Resolution Detection](https://openaccess.thecvf.com/content/CVPR2021/html/Qi_Multi-Scale_Aligned_Distillation_for_Low-Resolution_Detection_CVPR_2021_paper.html).” CVPR, 2021.
[^8]: Intel. “[OpenVINO Low Precision Transformations](https://docs.openvino.ai/2026/documentation/openvino-extensibility/openvino-plugin-library/advanced-guides/low-precision-transformations.html).” OpenVINO 2026 documentation.
[^9]: Tian, Zhi, et al. “[FCOS: Fully Convolutional One-Stage Object Detection](https://openaccess.thecvf.com/content_ICCV_2019/html/Tian_FCOS_Fully_Convolutional_One-Stage_Object_Detection_ICCV_2019_paper.html).” ICCV, 2019.
[^10]: Liu, Wei, et al. “[High-Level Semantic Feature Detection: A New Perspective for Pedestrian Detection](https://openaccess.thecvf.com/content_CVPR_2019/html/Liu_High-Level_Semantic_Feature_Detection_A_New_Perspective_for_Pedestrian_Detection_CVPR_2019_paper.html).” CVPR, 2019.
[^11]: Feng, Chengjian, et al. “[TOOD: Task-Aligned One-Stage Object Detection](https://openaccess.thecvf.com/content/ICCV2021/html/Feng_TOOD_Task-Aligned_One-Stage_Object_Detection_ICCV_2021_paper.html).” ICCV, 2021.
[^12]: Wang, Xinlong, et al. “[Repulsion Loss: Detecting Pedestrians in a Crowd](https://openaccess.thecvf.com/content_cvpr_2018/html/Wang_Repulsion_Loss_Detecting_CVPR_2018_paper.html).” CVPR, 2018.
[^13]: Bodla, Navaneeth, et al. “[Soft-NMS — Improving Object Detection With One Line of Code](https://openaccess.thecvf.com/content_iccv_2017/html/Bodla_Soft-NMS_--_Improving_ICCV_2017_paper.html).” ICCV, 2017.
[^14]: Liu, Songtao, Di Huang, and Yunhong Wang. “[Adaptive NMS: Refining Pedestrian Detection in a Crowd](https://openaccess.thecvf.com/content_CVPR_2019/html/Liu_Adaptive_NMS_Refining_Pedestrian_Detection_in_a_Crowd_CVPR_2019_paper.html).” CVPR, 2019.
[^15]: Zheng, Zhaohui, et al. “[Localization Distillation for Dense Object Detection](https://openaccess.thecvf.com/content/CVPR2022/html/Zheng_Localization_Distillation_for_Dense_Object_Detection_CVPR_2022_paper.html).” CVPR, 2022.
[^16]: Shao, Shuai, et al. “[CrowdHuman: A Benchmark for Detecting Human in a Crowd](https://arxiv.org/abs/1805.00123).” arXiv, 2018.
[^17]: Wang, Ao, et al. “[RepViT: Revisiting Mobile CNN From ViT Perspective](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_RepViT_Revisiting_Mobile_CNN_From_ViT_Perspective_CVPR_2024_paper.html).” CVPR, 2024.
[^18]: Qin, Danfeng, et al. “[MobileNetV4: Universal Models for the Mobile Ecosystem](https://arxiv.org/abs/2404.10518).” arXiv, 2024.
[^19]: Noh, Junhyug, et al. “[Improving Occlusion and Hard Negative Handling for Single-Stage Pedestrian Detectors](https://openaccess.thecvf.com/content_cvpr_2018/html/Noh_Improving_Occlusion_and_CVPR_2018_paper.html).” CVPR, 2018.
[^20]: Wang, Ao, et al. “[YOLOv10: Real-Time End-to-End Object Detection](https://arxiv.org/abs/2405.14458).” arXiv, 2024.
