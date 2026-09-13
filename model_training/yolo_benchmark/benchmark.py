"""Pretrained COCO YOLO nano models, evaluated with the project's CityPersons metrics."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time
import urllib.request

import cv2
import nncf
import numpy as np
import openvino as ov
import torch
from torchvision.ops import nms
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

from person_detection.data.dataset import CanonicalPersonDetectionDataset, letterbox_image
from person_detection.evaluation.metrics import MAPCalculator
from person_detection.optimization.pipeline import (
    IOU_THRESHOLDS, benchmark_core, compile_for_cpu, latency_summary,
    map_annotation_to_canvas, select_records,
)
from person_detection.training.pipeline import AzuriteClient, TrainingConfig

WEIGHTS_URLS = {
    'yolov8n': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt',
    'yolo26n': 'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt',
}


def positive_env(name, default):
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f'{name} must be positive')
    return value


def preprocess(rgb):
    canvas, _, _, _ = letterbox_image(rgb, 360, 640)
    return np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0


class CanvasModel(torch.nn.Module):
    """Keep the public 360x640 canvas; satisfy YOLO's stride inside the graph."""
    def __init__(self, detector):
        super().__init__()
        self.detector = detector

    def forward(self, images):
        return self.detector(torch.nn.functional.pad(images, (0, 0, 0, 24), value=0.0))


def decode(output, threshold=0.01, topk=1000, max_detections=100, *, end2end=False):
    output = np.asarray(output)
    if end2end:
        if output.ndim != 3 or output.shape[0] != 1 or output.shape[2] != 6:
            raise ValueError(f'Expected NMS-free output [1,N,6], got {output.shape}')
        rows = output[0]
        # Native top-300 across all COCO classes, then select person=0.
        rows = rows[np.isfinite(rows).all(axis=1) & (rows[:, 5] == 0) & (rows[:, 4] >= threshold)]
        rows = rows[np.argsort(-rows[:, 4], kind='stable')]
        boxes, scores = rows[:, :4].copy(), rows[:, 4].copy()
    else:
        if output.ndim != 3 or output.shape[:2] != (1, 84):
            raise ValueError(f'Expected YOLO COCO output [1,84,N], got {output.shape}')
        scores = output[0, 4]  # COCO person=0. Already probabilities, no sigmoid/objectness.
        indices = np.flatnonzero(np.isfinite(scores) & (scores >= threshold))
        indices = indices[np.argsort(-scores[indices], kind='stable')[:topk]]
        boxes = output[0, :4, indices].copy().reshape(-1, 4)
        scores = scores[indices].copy()
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, 640)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, 360)
    valid = np.isfinite(boxes).all(axis=1) & (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes, scores = boxes[valid], scores[valid]
    keep = (np.arange(min(len(scores), max_detections)) if end2end else
            nms(torch.from_numpy(boxes), torch.from_numpy(scores), 0.5)[:max_detections].numpy())
    return [{'box': boxes[i].tolist(), 'score': float(scores[i]), 'class': 1} for i in keep]


def verify_export(actual, reference, *, end2end=False):
    """Top-k may reorder near-tied detections; verify values after class/box matching."""
    if actual.shape != reference.shape:
        raise AssertionError(f'Export output shapes differ: {actual.shape} vs {reference.shape}')
    if end2end:
        a, b = actual[0], reference[0]
        cost = np.abs(a[:, None, :4] - b[None, :, :4]).sum(axis=2)
        cost += (a[:, None, 5] != b[None, :, 5]) * 10000
        rows, columns = linear_sum_assignment(cost)
        actual, reference = a[rows], b[columns]
        np.testing.assert_array_equal(actual[:, 5], reference[:, 5])
    np.testing.assert_allclose(actual, reference, rtol=0.002, atol=0.02,
                               err_msg='OpenVINO export differs from pretrained PyTorch output')
    return {'maxAbsoluteError': float(np.max(np.abs(actual-reference))),
            'comparison': 'class/box matched detections' if end2end else 'raw tensor'}


def evaluate(compiled, records, cache, *, end2end=False):
    calculator = MAPCalculator(IOU_THRESHOLDS, recall_fppi=0.1)
    for i, record in enumerate(records):
        tensor = preprocess(record.image_rgb(cache[record.sample['image']].read_bytes()))
        predictions = decode(compiled(tensor)[compiled.output(0)], end2end=end2end)
        boxes, ignores, metadata = map_annotation_to_canvas(record)
        calculator.add_image(i)
        calculator.add_predictions(i, predictions)
        calculator.add_ignore_regions(i, ignores)
        calculator.add_ground_truths(i, boxes, [1] * len(boxes), metadata)
        if (i + 1) % 50 == 0:
            print(f'Evaluated {i + 1}/{len(records)} images', flush=True)
    print('Calculating AP and size/visibility slices...', flush=True)
    return {'records': len(records), 'metrics': calculator.compute_map(verbose=False),
            'slices': calculator.compute_slices()}


def latency(compiled, records, cache, iterations, threshold, *, end2end=False):
    # Exclude storage/network access, but include image decode, resize, normalization and NMS.
    encoded = [cache[r.sample['image']].read_bytes() for r in records[:16]]
    sample = preprocess(records[0].image_rgb(encoded[0]))
    core = benchmark_core(compiled, sample, iterations)
    timings = []
    for i in range(iterations + 8):
        started = time.perf_counter()
        bgr = cv2.imdecode(np.frombuffer(encoded[i % len(encoded)], np.uint8), cv2.IMREAD_COLOR)
        tensor = preprocess(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        decode(compiled(tensor)[compiled.output(0)], threshold=threshold, end2end=end2end)
        if i >= 8:
            timings.append((time.perf_counter() - started) * 1000)
    return {'coreLatency': core, 'endToEndLatency': latency_summary(timings)}


def write_json(path, payload):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def main():
    model_name = os.getenv('YOLO_MODEL', 'yolov8n')
    if model_name not in WEIGHTS_URLS:
        raise ValueError(f'YOLO_MODEL must be one of {list(WEIGHTS_URLS)}')
    weights_url = WEIGHTS_URLS[model_name]
    end2end = model_name == 'yolo26n'
    out = Path(os.environ['YOLO_RUN_DIR'])
    out.mkdir(parents=True, exist_ok=True)
    threads = positive_env('OPENVINO_THREADS', 1)
    iterations = positive_env('BENCHMARK_ITERATIONS', 100)
    count = positive_env('VALIDATION_SAMPLES', 500)
    calibration_count = positive_env('CALIBRATION_SAMPLES', 300)
    quantize_setting = os.getenv('YOLO_INT8', 'true').lower()
    if quantize_setting not in ('true', 'false'):
        raise ValueError('YOLO_INT8 must be true or false')
    max_drop = float(os.getenv('MAX_ACCURACY_DROP', '0.01'))
    threshold = float(os.getenv('BENCHMARK_SCORE_THRESHOLD', '0.5'))
    if not 0 <= max_drop <= 1 or not 0 <= threshold <= 1:
        raise ValueError('Accuracy drop and score threshold must be between 0 and 1')
    torch.set_num_threads(threads)
    cv2.setNumThreads(threads)
    config = TrainingConfig(
        use_azurite=True,
        azurite_endpoint=os.getenv('AZURITE_BLOB_ENDPOINT', 'http://azurite:10000/devstoreaccount1'),
        azurite_access_key=os.getenv('AZURITE_ACCOUNT_NAME', 'devstoreaccount1'),
        azurite_secret_key=os.getenv('AZURITE_ACCOUNT_KEY', ''),
        azurite_connection_string=os.getenv('AZURITE_CONNECTION_STRING', ''),
        azurite_data_bucket=os.getenv('AZURITE_DATA_CONTAINER', 'computer-vision-data'),
    )
    client = AzuriteClient(config)
    if client.client is None:
        raise RuntimeError('Azurite connection failed; local fallback is disabled for this benchmark')
    dataset = CanonicalPersonDetectionDataset(client, split='val', input_height=360, input_width=640, augment=False)
    records = select_records(dataset, count, seed=1337, stratified=False)
    calibration = []
    if quantize_setting == 'true':
        train = CanonicalPersonDetectionDataset(client, split='train', input_height=360, input_width=640, augment=False)
        if any(train.dataset_metadata[k] != dataset.dataset_metadata[k]
               for k in ('versionPrefix', 'manifestSha256', 'schemaVersion')):
            raise RuntimeError('Dataset current pointer changed while loading splits; retry')
        calibration = select_records(train, calibration_count, seed=1337, stratified=True)
    cache_dir = Path(os.getenv('YOLO_STATE_ROOT', '/state')) / 'image-cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = {}
    for record in records + calibration:
        path = cache_dir / record.sample['image_sha256']
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != path.name:
            path.write_bytes(record.image_bytes())
        cache[record.sample['image']] = path
    weights = Path(os.getenv('YOLO_STATE_ROOT', '/state')) / 'weights' / f'{model_name}.pt'
    weights.parent.mkdir(parents=True, exist_ok=True)
    if not weights.exists():
        print(f'Downloading {weights_url}', flush=True)
        temp = weights.with_suffix('.download')
        urllib.request.urlretrieve(weights_url, temp)
        temp.replace(weights)
    yolo = YOLO(str(weights), task='detect')
    if yolo.names.get(0) != 'person' or len(yolo.names) != 80:
        raise RuntimeError('Expected original 80-class COCO checkpoint')
    detector = yolo.model.cpu().float().eval()
    head = detector.model[-1]
    if bool(head.end2end) != end2end:
        raise RuntimeError(f'Unexpected detection head mode for {model_name}')
    head.export, head.format, head.xyxy = True, 'openvino', True
    if end2end:
        head.max_det = 300
        head.fuse()  # Remove the unused one-to-many training branch, keep native one-to-one inference.
    wrapper = CanvasModel(detector).eval()
    sample = preprocess(records[0].image_rgb(cache[records[0].sample['image']].read_bytes()))
    with torch.no_grad():
        reference = wrapper(torch.from_numpy(sample)).numpy()
        model = ov.convert_model(wrapper, example_input=torch.from_numpy(sample), input=[1, 3, 360, 640])
    fp32_path = out / f'{model_name}_fp32.xml'
    ov.save_model(model, fp32_path, compress_to_fp16=False)
    compiled = compile_for_cpu(fp32_path, threads)
    actual = compiled(sample)[compiled.output(0)]
    parity = verify_export(actual, reference, end2end=end2end)
    report = {
        'status': 'running', 'model': model_name, 'fineTuned': False,
        'weights': {'url': weights_url, 'sha256': hashlib.sha256(weights.read_bytes()).hexdigest()},
        'dataset': dataset.dataset_metadata,
        'validationSelection': [r.identity() for r in records],
        'calibrationSelection': [r.identity() for r in calibration],
        'preprocessing': {'canvasWH': [640, 360], 'internalWH': [640, 384], 'padding': 'zero; extra 24 bottom rows in graph',
                          'input': 'NCHW RGB float32 /255; aspect-preserving letterbox'},
        'evaluation': {'metricImplementation': 'project MAPCalculator, all-point interpolated AP; not pycocotools',
                       'personClass': 0, 'scoreThreshold': 0.01, 'nmsIoU': None if end2end else 0.5,
                       'postprocessing': 'native NMS-free' if end2end else 'person NMS',
                       'nativeTopKAllClasses': 300 if end2end else None,
                       'preNmsTopK': None if end2end else 1000, 'maxDetections': 100, 'recallFPPI': 0.1,
                       'fullValidationSplit': len(records) == len(dataset)},
        'runtime': {'backend': 'OpenVINO CPU', 'threads': threads, 'streams': 1, 'precisionHint': 'f32',
                    'platform': platform.platform(), 'cpu': platform.processor(), 'loadAverage': os.getloadavg(),
                    'versions': {p: importlib.metadata.version(p) for p in ['ultralytics', 'torch', 'openvino', 'nncf']},
                    'latencyScoreThreshold': threshold, 'warmupIterations': 8,
                    'endToEndScope': 'cached encoded image decode through person postprocessing; excludes storage/network'},
        'exportParity': parity,
        'results': {},
    }
    for precision in ['fp32'] + (['int8'] if calibration else []):
        if precision == 'int8':
            def transform(record):
                return preprocess(record.image_rgb(cache[record.sample['image']].read_bytes()))
            quantized = nncf.quantize(model, nncf.Dataset(calibration, transform),
                                      subset_size=len(calibration), preset=nncf.QuantizationPreset.MIXED,
                                      target_device=nncf.TargetDevice.CPU)
            path = out / f'{model_name}_int8.xml'
            ov.save_model(quantized, path, compress_to_fp16=False)
            compiled = compile_for_cpu(path, threads)
        print(f'Evaluating and timing {precision}', flush=True)
        result = evaluate(compiled, records, cache, end2end=end2end)
        result.update(latency(compiled, records, cache, iterations, threshold, end2end=end2end))
        report['results'][precision] = result
        write_json(out / 'report.json', report)
    if calibration:
        drop = report['results']['fp32']['metrics']['mAP@0.50:0.95'] - report['results']['int8']['metrics']['mAP@0.50:0.95']
        report['int8AccuracyGate'] = {'absoluteDrop': drop, 'maximumDrop': max_drop, 'passed': drop <= max_drop}
    report['status'] = 'complete'
    write_json(out / 'report.json', report)
    lines = [f'# Pretrained {model_name} benchmark', '',
             f'Validation images: {len(records)}. Canvas: 640×360; internal padding: 640×384.', '',
             '| Precision | AP50 | AP50:95 | Recall @ FPPI 0.1 | Core mean ms | E2E mean ms |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for precision, result in report['results'].items():
        metrics = result['metrics']
        lines.append(f"| {precision} | {metrics['mAP@0.50']:.4f} | {metrics['mAP@0.50:0.95']:.4f} | "
                     f"{metrics['Recall@FPPI=0.10']:.4f} | {result['coreLatency']['meanMs']:.2f} | "
                     f"{result['endToEndLatency']['meanMs']:.2f} |")
    if calibration:
        lines.extend(['', 'INT8 accuracy gate: ' + json.dumps(report['int8AccuracyGate'])])
    lines.extend(['', 'Timings reflect current host load; compare with other compute jobs idle.', ''])
    summary = '\n'.join(lines)
    (out / 'summary.md').write_text(summary)
    print(summary)
    print(f'Benchmark saved: {out / "report.json"}')


if __name__ == '__main__':
    main()
