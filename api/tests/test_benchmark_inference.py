"""Exercise load pacing and reporting through a real local HTTP server."""
from argparse import Namespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

from benchmark_inference import run, summary


class BenchmarkTests(unittest.TestCase):
    def test_percentiles_and_no_successes(self):
        self.assertEqual(summary([]), {'samples': 0})
        self.assertEqual(summary(list(range(1, 101)))['p95Ms'], 95)

    def test_bounded_camera_load_reports_busy_and_missed_slots(self):
        lock = threading.Lock()
        counts = {'requests': 0, 'active': 0, 'peak': 0}

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']))
                with lock:
                    counts['requests'] += 1
                    sequence = counts['requests']
                    counts['active'] += 1
                    counts['peak'] = max(counts['peak'], counts['active'])
                time.sleep(.05)
                status = 429 if sequence % 3 == 0 else 200
                body = json.dumps({'inferenceMs': 10, 'releaseId': 'fixture', 'timings': {'queueMs': 2}}).encode()
                self.send_response(status)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                with lock:
                    counts['active'] -= 1

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                frame = Path(directory) / 'frame.jpg'
                frame.write_bytes(b'encoded fixture')
                args = Namespace(images=[str(frame)], url=[f'fixture=http://127.0.0.1:{server.server_port}'],
                                 iterations=4, warmups=1, cameras=3, fps=60, timeout=2,
                                 threshold=.5, hardware_notes='test server', output=str(Path(directory) / 'report.json'))
                report = run(args)['results']['fixture']
                self.assertEqual(sum(report['responses'].values()), 12)
                self.assertEqual(report['responses']['429'], 4)
                self.assertEqual(report['timings']['httpSuccess']['samples'], 8)
                self.assertGreater(report['missedCaptureSlots'], 0)
                self.assertLessEqual(counts['peak'], 3)
                self.assertEqual(len(report['perCamera']), 3)
                self.assertEqual(report['releaseIds'], ['fixture'])
                self.assertEqual(counts['requests'], 13)  # warmup excluded, no busy retries
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    unittest.main()
