"""Compare complete HTTP inference paths on identical encoded frames.
Usage: python benchmark_inference.py --url csharp=http://localhost:8080 \
    --images frame1.jpg frame2.png --output benchmark.json
Use equivalent native CPU threads/streams and idle hardware; no training concurrently.
"""
import argparse
import json
import statistics
import time
import urllib.request
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--url',action='append',required=True,help='label=http://host:port')
p.add_argument('--images',nargs='+',required=True)
p.add_argument('--output',required=True)
p.add_argument('--iterations',type=int,default=100)
p.add_argument('--warmups',type=int,default=8)
p.add_argument('--threshold',type=float,default=.5)
a=p.parse_args()
if a.iterations<1 or a.warmups<0: p.error('Invalid iteration counts')
boundary='camera-benchmark-boundary'
bodies=[]
for name in a.images:
    bodies.append((f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="frame"\r\n'
                   'Content-Type: application/octet-stream\r\n\r\n').encode()+Path(name).read_bytes()+
                  f'\r\n--{boundary}--\r\n'.encode())

def summary(values):
    s=sorted(values)
    return {'meanMs':statistics.mean(s),'p50Ms':statistics.median(s),
            'p95Ms':s[min(len(s)-1,int(.95*(len(s)-1)))],'samples':len(s)}
report={'images':a.images,'iterations':a.iterations,'warmups':a.warmups,'threshold':a.threshold,
        'scope':'Encoded-image HTTP request through response body; excludes file read and multipart construction',
        'results':{}}
for setting in a.url:
    label,url=setting.split('=',1)
    timings={}
    for i in range(a.warmups+a.iterations):
        req=urllib.request.Request(f'{url.rstrip("/")}/api/inference/detect?threshold={a.threshold}',
                                   data=bodies[i%len(bodies)],headers={'Content-Type':f'multipart/form-data; boundary={boundary}'})
        start=time.perf_counter()
        with urllib.request.urlopen(req,timeout=60) as response: payload=response.read()
        elapsed=(time.perf_counter()-start)*1000
        result=json.loads(payload)
        if i<a.warmups: continue
        timings.setdefault('httpTotal',[]).append(elapsed)
        timings.setdefault('reportedInference',[]).append(result['inferenceMs'])
        for key,value in result.get('timings',{}).items(): timings.setdefault(key,[]).append(value)
    report['results'][label]={k:summary(v) for k,v in timings.items()}
Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
