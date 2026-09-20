"""Warm, paced multi-camera HTTP benchmark. Run on idle deployment hardware.

Each worker represents one camera, never queues frames, and counts HTTP 429 as a
frame drop. Frame sources must be representative camera JPEG/PNG files. HTTP
latency excludes RTSP/HLS, browser decode, JPEG encoding and rendering.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import time
import urllib.error
import urllib.request


def summary(values):
    if not values:
        return {"samples": 0}
    ordered = sorted(values)
    return {"meanMs": statistics.mean(ordered), "p50Ms": statistics.median(ordered),
            "p95Ms": ordered[math.ceil(.95 * len(ordered)) - 1],
            "p99Ms": ordered[math.ceil(.99 * len(ordered)) - 1], "samples": len(ordered)}


def run(args):
    boundary = 'camera-benchmark-boundary'
    bodies = [(f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="frame"\r\n'
               'Content-Type: application/octet-stream\r\n\r\n').encode() + Path(name).read_bytes() +
              f'\r\n--{boundary}--\r\n'.encode() for name in args.images]
    report = {"createdAt": datetime.now(timezone.utc).isoformat(), "images": args.images,
              "iterationsPerCamera": args.iterations, "warmups": args.warmups,
              "cameras": args.cameras, "fpsPerCamera": args.fps, "threshold": args.threshold,
              "hardwareNotes": args.hardware_notes,
              "scope": "Encoded-frame HTTP round trip; excludes RTSP/HLS, decode, encoding and rendering",
              "results": {}}
    for setting in args.url:
        label, url = setting.split('=', 1)

        def request(index):
            req = urllib.request.Request(f'{url.rstrip("/")}/api/inference/detect?threshold={args.threshold}',
                                         data=bodies[index % len(bodies)],
                                         headers={'Content-Type': f'multipart/form-data; boundary={boundary}'})
            start = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=args.timeout) as response:
                    payload = json.loads(response.read())
                return 200, (time.perf_counter() - start) * 1000, payload
            except urllib.error.HTTPError as error:
                error.close()
                return error.code, (time.perf_counter() - start) * 1000, None
            except (OSError, ValueError):
                return 'transportOrPayloadError', (time.perf_counter() - start) * 1000, None

        for i in range(args.warmups):
            status, _, _ = request(i)
            if status != 200:
                raise RuntimeError(f'{label}: warmup failed with {status}')

        def camera(camera_index):
            counts, timings, release_ids = {}, {}, set()
            missed = 0
            deadline = time.perf_counter()
            for i in range(args.iterations):
                time.sleep(max(0, deadline - time.perf_counter()))
                status, elapsed, payload = request(i + camera_index)
                key = str(status)
                counts[key] = counts.get(key, 0) + 1
                timings.setdefault('httpAll', []).append(elapsed)
                if status == 200:
                    timings.setdefault('httpSuccess', []).append(elapsed)
                    timings.setdefault('reportedInference', []).append(payload['inferenceMs'])
                    release_ids.add(payload.get('releaseId'))
                    for key, value in payload.get('timings', {}).items():
                        timings.setdefault(key, []).append(value)
                deadline += 1 / args.fps
                # Missed capture slots are dropped, never replayed in a burst.
                now = time.perf_counter()
                if deadline < now:
                    skipped = math.ceil((now - deadline) * args.fps)
                    missed += skipped
                    deadline += skipped / args.fps
            return counts, timings, missed, release_ids

        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.cameras) as pool:
            runs = list(pool.map(camera, range(args.cameras)))
        elapsed = time.perf_counter() - started
        counts, timings, releases = {}, {}, set()
        for camera_counts, camera_timings, _, camera_releases in runs:
            for key, value in camera_counts.items():
                counts[key] = counts.get(key, 0) + value
            for key, values in camera_timings.items():
                timings.setdefault(key, []).extend(values)
            releases.update(camera_releases)
        report['results'][label] = {
            'durationSeconds': elapsed, 'responses': counts,
            'successfulFramesPerSecond': counts.get('200', 0) / elapsed,
            'missedCaptureSlots': sum(item[2] for item in runs),
            'releaseIds': sorted(releases, key=str),
            'timings': {key: summary(values) for key, values in timings.items()},
            'perCamera': [{'responses': item[0], 'missedCaptureSlots': item[2],
                           'timings': {key: summary(values) for key, values in item[1].items()}}
                          for item in runs]}
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', action='append', required=True, help='label=http://host:port')
    parser.add_argument('--images', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--warmups', type=int, default=8)
    parser.add_argument('--cameras', type=int, default=4)
    parser.add_argument('--fps', type=float, default=2)
    parser.add_argument('--timeout', type=float, default=10)
    parser.add_argument('--threshold', type=float, default=.5)
    parser.add_argument('--hardware-notes', required=True, help='CPU, thread settings, idle/load conditions')
    args = parser.parse_args()
    if args.iterations < 1 or args.warmups < 0 or not 1 <= args.cameras <= 64:
        parser.error('Invalid iteration, warmup or camera count')
    if not 0 < args.fps <= 60 or not 0 < args.timeout <= 120 or not 0 <= args.threshold <= 1:
        parser.error('Invalid FPS, timeout or threshold')
    if any('=' not in setting or not setting.split('=', 1)[1].startswith(('http://', 'https://'))
           for setting in args.url):
        parser.error('--url must be label=http(s)://host:port')
    print(json.dumps(run(args), indent=2))


if __name__ == '__main__':
    main()
