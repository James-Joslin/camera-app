"""HTTP correctness regression against generate_inference_fixtures.py output.
Usage: python verify_inference.py FIXTURE_DIR URL PRECISION
Run API once per precision with MODEL_PATH selecting the fixture XML.
"""
import concurrent.futures
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
root,base,precision = Path(sys.argv[1]),sys.argv[2].rstrip('/'),sys.argv[3]

def send(data,threshold=.5):
    boundary = 'camera-validation-boundary'
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="frame.png"\r\n'
            'Content-Type: application/octet-stream\r\n\r\n').encode()+data+f'\r\n--{boundary}--\r\n'.encode()
    request = urllib.request.Request(f'{base}/api/inference/detect?threshold={threshold}',data=body,
               headers={'Content-Type':f'multipart/form-data; boundary={boundary}'})
    try:
        with urllib.request.urlopen(request,timeout=60) as response:
            return response.status,json.load(response)
    except urllib.error.HTTPError as e:
        return e.code,json.load(e)

cases = [x for x in json.loads((root/'expected.json').read_text()) if x['precision']==precision]
assert cases, f"No fixtures for {precision}"
for case in cases:
    code,result = send((root/case['image']).read_bytes(),case['threshold'])
    assert code == 200,(code,result)
    assert result['image'] == {'width':case['width'],'height':case['height']}
    assert len(result['detections']) == len(case['detections']),(case,result)
    for actual,expected in zip(result['detections'],case['detections']):
        assert actual['label']==expected['label'] and actual['classId']==expected['classId']
        assert abs(actual['confidence']-expected['confidence']) < 2e-5,(actual,expected)
        assert max(abs(a-b) for a,b in zip(actual['box'],expected['box'])) <= 1,(actual,expected)
    assert all(x>=0 for x in result['timings'].values())
    assert abs(result['inferenceMs']-result['timings']['inferenceMs']) <= .006
assert send(b'not an image')[0] == 400
assert send((root/'odd.png').read_bytes(),1.5)[0] == 400
# Multiple concurrent requests exercise worker isolation and backpressure.
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    results = list(executor.map(lambda _:send((root/'odd.png').read_bytes()),range(16)))
assert all(code in (200,429) for code,_ in results),results
assert any(code==200 for code,_ in results)
for code,result in results:
    if code==200: assert result['image']=={'width':777,'height':333}
print(f'PASS: {len(cases)} {precision} parity cases, invalid image/threshold, concurrent requests')
