"""Run with the training environment; produce synthetic contract/parity fixtures.
Usage: python generate_inference_fixtures.py OUTPUT_DIR [PYTHON_MODEL_RUNTIME]
No training or production models are modified.
"""
import importlib.util
import json
import sys
from pathlib import Path
import cv2
import numpy as np
import openvino as ov
from openvino import opset13 as op

root = Path(sys.argv[1]); root.mkdir(parents=True, exist_ok=True)
reference_path = sys.argv[2] if len(sys.argv) > 2 else str(Path(__file__).with_name("reference_runtime.py"))
spec = importlib.util.spec_from_file_location("reference", reference_path)
reference = importlib.util.module_from_spec(spec); spec.loader.exec_module(reference)
image = op.parameter([1, 3, 360, 640], np.float32)
# Make scores depend on every normalized input channel, so bad preprocessing fails parity.
weighted = op.convolution(image, op.constant(np.array([.2,.3,.5],np.float32).reshape(1,3,1,1)),
                          [1,1],[0,0],[0,0],[1,1])
mean = op.reduce_mean(weighted, op.constant(np.array([0, 1, 2, 3], np.int64)), False)
base = np.array([4, 3, 2, -4, 1, .4, -.4, -2], np.float32).reshape(1,8,1)
logits = op.add(op.constant(base), op.multiply(mean,op.constant(np.float32(.1))))
coords = np.array([[20,30,120,280],[22,32,118,278],[200,20,260,200],
                   [350,50,420,300],[-20,-10,50,100],[20,20,20,30],
                   [400,100,700,400],[300,0,330,40]], np.float32).reshape(1,8,4)
boxes = op.constant(coords)
logits.output(0).set_names({'person_logits'})
boxes.output(0).set_names({'boxes_xyxy_pixels'})
model = ov.Model([logits,boxes],[image],"preprocessing-contract")
for precision in ('fp32','fp16'):
    ov.save_model(model, root / f'person_detector_{precision}.xml', compress_to_fp16=precision=='fp16')
# A model with incompatible outputs must fail explicitly instead of silently misdecoding.
legacy = ov.Model([op.constant(base),op.constant(coords)],[image])
ov.save_model(legacy,root/'incompatible.xml',compress_to_fp16=False)
# This INT8 graph is a transport/runtime test, not a deployable calibrated detector.
import nncf
calibration = [np.random.default_rng(i).normal(size=(1,3,360,640)).astype(np.float32) for i in range(2)]
quantized = nncf.quantize(model, nncf.Dataset(calibration), subset_size=2)
ov.save_model(quantized,root/'person_detector_int8.xml',compress_to_fp16=False)
rng = np.random.default_rng(17)
fixtures = []
for name,shape in [('landscape',(720,1280,3)),('portrait',(479,321,3)),('odd',(333,777,3))]:
    frame = rng.integers(0,256,shape,dtype=np.uint8)
    frame[:,:,0] //= 4
    frame[:,:,2] = 192 + frame[:,:,2] // 4
    for extension in ('png','jpg'):
        path = root/f'{name}.{extension}'
        cv2.imwrite(str(path),frame)
        decoded = cv2.imread(str(path))
        for precision in ('fp32','fp16','int8'):
            runtime = reference.OpenVinoPersonDetector(root/f'person_detector_{precision}.xml')
            for threshold in (.01,.5,.99):
                detections,_ = runtime.predict(decoded,threshold)
                fixtures.append({'image':path.name,'precision':precision,'threshold':threshold,
                                 'width':shape[1],'height':shape[0],'detections':detections})
(root/'expected.json').write_text(json.dumps(fixtures,indent=2))
print(f'Created {len(fixtures)} reference cases in {root}')
