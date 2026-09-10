# Person Detector Architecture Improvement Strategy

## Executive recommendation

The next production architecture should remain a compact convolutional one-stage detector, but it should stop spending compute on square padding, dense FPN convolutions, redundant normalization, and six-to-nine anchors per location.

The recommended progression is:

1. Establish a valid, measured baseline for the current quality-aware model.
2. Make the input canvas rectangular and configurable, starting with 640×320 for CityPersons and 640×360 for a 16:9 camera.
3. Replace the dense top-down FPN smoothing convolutions with one lightweight depthwise-separable, weighted bidirectional feature-fusion block.
4. Simplify and share the detection tower: one normalization/activation sequence per convolution, no default CBAM-style attention, shared tower weights across pyramid levels, and small level-specific output layers.
5. Build an anchor-free pedestrian variant with one prediction per feature-map location, direct l/t/r/b regression, and the existing joint person/localization-quality score.
6. Improve crowd behavior with training-only repulsion loss and a Soft-NMS experiment before adding a learned density or occlusion subnetwork.
7. Use a larger or higher-resolution model as a training teacher; keep the deployed student compact.

A transformer or NMS-free detector is not the next move. The dataset is small, the current quality-head model has not yet produced a valid baseline checkpoint, and the existing architecture has simpler, measurable inefficiencies.

## Scope and constraints

This assessment assumes:

- Binary person detection from fixed cameras.
- CityPersons as the current supervised dataset: 2,975 training images and 500 validation images.
- A 480×480 current input, CPU inference through OpenVINO, and manifest-driven INT8 calibration.
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

The model taps MobileNetV3-Small at three stages, adds stride-64 and stride-128 features, applies a 128-channel top-down FPN, and predicts on five levels. At 480×480, those levels contain 4,805 spatial locations but emit 29,070 anchors:

| Level | Locations | Anchors/location | Predictions |
| --- | ---: | ---: | ---: |
| stride 8 | 3,600 | 6 | 21,600 |
| stride 16 | 900 | 6 | 5,400 |
| stride 32 | 225 | 6 | 1,350 |
| stride 64 | 64 | 9 | 576 |
| stride 128 | 16 | 9 | 144 |
| Total | 4,805 | — | 29,070 |

The current output shapes are `[B, 29070, 1]` for the joint person-quality logit and `[B, 29070, 4]` for box offsets.

### Strengths worth retaining

- A stride-8 prediction level for small pedestrians.
- Multi-scale feature fusion. FPN established the value of semantically strong high-resolution features for scale-varying detection.[^2]
- ATSS assignment. Its central finding is that sample selection matters more than the nominal anchor/anchor-free distinction, and it adds no inference overhead.[^3]
- A single joint person/localization-quality logit. This follows the consistency argument behind Generalized Focal Loss: the ranking score should express both classification and localization quality.[^4]
- GIoU regression, ignore-region neutralization, hard-case sampling, and immutable metric provenance.
- A conventional convolutional operator set that is suitable for OpenVINO INT8.

### Weaknesses visible in the code

1. **The square canvas wastes spatial budget.** CityPersons frames are 2048×1024. Letterboxing them into 480×480 produces 480×240 image content plus 120 pixels of padding above and below. Half of the tensor contains no scene information, while a 50-pixel source pedestrian becomes only about 11.7 pixels tall.

2. **The neck is disproportionately expensive.** Five dense 3×3, 128-to-128 FPN smoothing convolutions contain 737,280 kernel weights and require about 709 million multiply-accumulates across the five maps. A depthwise 3×3 plus pointwise 1×1 replacement would use about 87,680 kernel weights and 84 million multiply-accumulates for the same maps, before normalization: roughly an 8.4× reduction for this part of the neck. Actual OpenVINO latency must still be measured because operator efficiency is hardware-specific.

3. **The head stacks normalization and nonlinearities redundantly.** Each classification and regression subnet calls a separable block that already contains BatchNorm and LeakyReLU, then immediately applies GroupNorm and SiLU. This creates two normalizations and two activations around one feature transform. It is difficult to justify statistically, complicates quantization, and prevents a clean Conv–BatchNorm folding path.

4. **Every level has an independent head.** Five separate towers consume parameters and allow small CityPersons training splits to learn inconsistent level-specific features. The final output convolutions must remain level-specific while anchor counts differ, but most tower weights can be shared.

5. **The attention path is branch-heavy.** Every pyramid level performs average pooling, max pooling, two channel gates, channel reductions, a 7×7 spatial convolution, and sigmoid gates. These operations may help accuracy, but they are not free merely because parameter counts are small. MobileOne’s deployment study is a useful warning that FLOPs and parameter counts can correlate poorly with real latency, and that branching and activation choices matter.[^5]

6. **The FPN is top-down and unweighted only.** All lateral inputs are summed equally, and information does not travel back from high-resolution maps toward lower-resolution maps. EfficientDet found that learnable weighted, bidirectional fusion can improve the accuracy/efficiency frontier.[^6]

7. **Anchor multiplication is no longer clearly earning its cost.** The ATSS paper itself reports that tiling multiple anchors per location may not be necessary once sample assignment is strong.[^3] Here it multiplies candidates by six or nine, increasing final convolution width, target assignment, decoding, thresholding, and NMS.

8. **Crowd structure is addressed mainly by ignore masks.** Fixed hard NMS at IoU 0.5 can suppress two nearby pedestrians, and ordinary GIoU does not explicitly prevent a box from drifting toward an adjacent person.

9. **Visible-box supervision is unused by the model.** Visible boxes affect augmentation and metadata slices, but there is no training signal that asks the feature extractor to represent visible extent or occlusion. This is an opportunity only after the visible-box data contract is corrected and exhaustively validated.

## Highest-priority improvements

## 1. Rectangular, deployment-shaped input

Make `input_height` and `input_width` independent throughout preprocessing, anchor generation, decoding, export, calibration, and evaluation.

Two useful starting points are:

| Canvas | Pixels | CityPersons content | Padding | Scale from 2048×1024 |
| --- | ---: | ---: | ---: | ---: |
| Current 480×480 | 230,400 | 480×240 | 50.0% | 0.234× |
| 640×320 | 204,800 | 640×320 | 0% | 0.3125× |
| 640×360 | 230,400 | 640×320 | 11.1% | 0.3125× |
| 768×384 | 294,912 | 768×384 | 0% | 0.375× |

At 640×320, a 50-pixel source pedestrian becomes 15.6 pixels tall rather than 11.7 pixels, a 33% linear increase, while total input pixels fall by 11%. At 640×360, the pixel count exactly matches 480×480 and leaves room for a 16:9 production stream.

This is an analytical projection, not an accuracy claim. It is nevertheless the cleanest first experiment because it reallocates existing computation from padding to people. Low-resolution detection research also supports the importance of preserving spatial information and shows that multi-resolution teachers can improve low-resolution students without changing student inference cost.[^7]

Implementation implications:

- Store input shape as `(height, width)`, not one scalar.
- Generate anchors from actual per-level `(H, W)` feature shapes.
- Normalize x coordinates by width and y coordinates by height.
- Export OpenVINO with the chosen fixed rectangular shape.
- Use the production camera aspect ratio for release testing, not only the dataset ratio.
- Keep stride 8 initially. Add a stride-4 head only if the rectangular baseline still has poor Reasonable-small miss rate; stride 4 is expensive.

**Recommendation:** make 640×320 the first CityPersons experiment and 640×360 the first deployment-shaped experiment.

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

At the current feature sizes, one prediction per location reduces candidates from 29,070 to 4,805—an 83.5% reduction. It also removes hand-designed scales and aspect ratios. FCOS established direct per-location l/t/r/b regression as a strong anchor-free design,[^9] while pedestrian-specific CSP showed that center-and-scale prediction can be both structurally simple and competitive on pedestrian benchmarks.[^10]

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

## Proposed target architecture

```text
RGB frame
  ↓
configurable rectangular letterbox (640×320 or camera-shaped 640×360)
  ↓
MobileNetV3-Small control backbone
  ├── stride 8
  ├── stride 16
  └── stride 32
  ↓
lightweight stride-64 and stride-128 blocks
  ↓
one 96/128-channel separable weighted BiFPN pass
  ↓
shared decoupled separable towers
  ├── joint person × IoU score: 1 value/location
  ├── l/t/r/b box distances: 4 values/location
  └── visibility auxiliary: training only
  ↓
decode → score floor → Soft-NMS/hard-NMS experiment → detections
```

This target preserves the current strengths—P2-scale detail, quality-aware ranking, ignore regions, GIoU, and INT8-friendly convolution—while removing square-padding waste, dense neck convolutions, redundant normalization, duplicated towers, and anchor multiplication.

## Experiment roadmap

Never combine all changes into one training run. The current CPU training time is about 20 minutes per epoch, so poorly isolated experiments are expensive and uninterpretable.

| Order | Experiment | Change from control | Primary question |
| ---: | --- | --- | --- |
| 0 | B0 | Valid current quality-head model at 480×480 | What is the real accuracy/latency baseline? |
| 1 | R1 | 640×320 input only | Does spatial reallocation improve small/heavy metrics without latency cost? |
| 2 | R2 | 640×360 input only | Does the camera-shaped canvas give the best operational trade-off? |
| 3 | N1 | Separable FPN smoothing only | How much neck latency/size can be removed safely? |
| 4 | N2 | One separable weighted BiFPN pass | Does bidirectional fusion recover or improve accuracy? |
| 5 | H1 | Clean shared head, no attention | Are double normalization and per-level towers unnecessary? |
| 6 | H2 | Lightweight attention added back | Does attention provide measurable value after cleanup? |
| 7 | C1 | Repulsion Loss | Does heavy-occlusion miss rate improve without runtime cost? |
| 8 | C2 | Soft-NMS | Are crowded-person misses caused by suppression? |
| 9 | A1 | Anchor-free direct regression | Can candidate count fall 83.5% without recall loss? |
| 10 | A2 | Task-aligned assignment | Does classification/localization alignment improve ranking? |
| 11 | D1 | High-resolution teacher distillation | Can the compact student recover localization accuracy? |
| 12 | K1 | Backbone finalists | Which backbone wins on the actual OpenVINO CPU? |

Use one seed for screening and at least three seeds for finalists because 2,975 training images make small metric differences noisy. Preserve the same manifest, augmentation policy, schedule, evaluation thresholds, and INT8 calibration records.

## Acceptance gates

Every architecture variant should report:

- Project AP50 and AP50:95.
- Recall at FPPI 0.10.
- Small, medium, large, clear, partially occluded, and heavily occluded slices.
- Official Reasonable, Reasonable-small, heavy-occlusion, and All log-average miss rate.
- FP32/FP16/INT8 model bytes.
- OpenVINO core and end-to-end p50/p95 latency at fixed thread count.
- INT8 accuracy drop against that variant’s own FP32 result.
- Parameters, MACs, decoded candidate count, and peak preprocessing/inference memory.

Suggested screening rules:

- A simplification is successful if it reduces end-to-end p50 latency by at least 10% or model size by at least 15%, while worsening official Reasonable miss rate by no more than 0.3 percentage points and AP50:95 by no more than 0.5 points.
- An accuracy feature is successful if it improves heavy-occlusion or Reasonable-small miss rate by at least 1 percentage point without worsening end-to-end p95 latency by more than 10%.
- A backbone is acceptable only after successful OpenVINO FP16 and INT8 export and the existing maximum 0.01 absolute AP50:95 INT8-drop gate.
- A result that appears only at one seed should be treated as provisional.

These thresholds are engineering recommendations, not published standards. They should be adjusted when an explicit camera FPS and memory budget is defined.

## Baseline warning

The existing `optimized/benchmark.json` reports historical FP16 mean latency of 8.29 ms and INT8 mean latency of 4.69 ms. It should not be used as the current architecture baseline. Safe checkpoint metadata inspection shows that `best_model_fp32.pth` has no `modelFormatVersion` or dataset provenance and has 12/18 classification output channels at the five levels—the legacy two-logit softmax head. The current model would have 6/9 output channels for its one-logit head.

The quality-head training run failed before its first validation checkpoint was saved. Architecture decisions should therefore wait for a valid model-format-v2 checkpoint and complete accuracy-controlled optimization report.

## What not to prioritize

- **A large transformer or DETR conversion:** too much toolchain and data risk before simpler inefficiencies are removed.
- **More attention blocks:** the current architecture already has attention at every level without evidence that it earns latency.
- **A stride-4 prediction head immediately:** likely useful only if rectangular input and improved fusion still leave small-person recall deficient.
- **Distributional regression immediately:** credible for localization, but it increases output width; establish the anchor-free direct-regression baseline first.
- **NMS-free dual-assignment training immediately:** methods such as YOLOv10 target end-to-end latency,[^20] but add assignment and training complexity that is premature for this small binary dataset.
- **Changing backbone, neck, head, assignment, loss, and input shape together:** it prevents causal measurement and makes INT8 regressions difficult to diagnose.

## Conclusion

The strongest near-term architecture is not “MobileNetV3 plus more modules.” It is a spatially efficient, rectangular, lightweight pedestrian detector.

The first production-minded prototype should use 640×320 or 640×360 input, MobileNetV3-Small, a single separable weighted BiFPN pass, a clean shared head, and the current ATSS plus joint quality objective. This can be implemented incrementally and benchmarked against the existing design.

The next-generation branch should then replace 29,070 anchors with about 4,805 point predictions and direct l/t/r/b regression. Crowd-specific loss, Soft-NMS, and teacher distillation should be layered around that baseline only when the official small/heavy-occlusion metrics demonstrate the need.

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
