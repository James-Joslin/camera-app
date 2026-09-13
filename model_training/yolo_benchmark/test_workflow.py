"""Exercise launcher branches without downloading or publishing a dataset."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class WorkflowTests(unittest.TestCase):
    def run_workflow(self, mode, run_id='test'):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / 'state'
            state.mkdir()
            script = root / 'entrypoint.sh'
            script.write_text(Path(__file__).with_name('entrypoint.sh').read_text().replace(
                'ROOT=/opt/model_training', f'ROOT="{root}"'))
            python = root / 'python'
            python.write_text('''#!/usr/bin/env bash
set -eu
if [[ "$*" == *validate_citypersons_azurite* ]]; then
    echo validate >> "$CALL_LOG"
    if [[ "$TEST_MODE" == missing && ! -f "$BOOTSTRAP_MARKER" || "$TEST_MODE" == broken ]]; then exit 1; fi
else
    echo benchmark >> "$CALL_LOG"
fi
''')
            bootstrap = root / 'getCityPersons.sh'
            bootstrap.write_text('''#!/usr/bin/env bash
set -eu
[[ "$CITYPERSONS_DATASET_VERSION" =~ ^v[0-9]{4}-[0-9]{2}-[0-9]{2}([._-][A-Za-z0-9]+)*$ ]] || exit 2
echo bootstrap >> "$CALL_LOG"
touch "$BOOTSTRAP_MARKER"
''')
            python.chmod(0o755)
            bootstrap.chmod(0o755)
            log = root / 'calls'
            environment = {**os.environ, 'PATH': f'{root}:{os.environ["PATH"]}',
                           'YOLO_STATE_ROOT': str(state), 'YOLO_RUN_ID': run_id,
                           'CITYPERSONS_DATASET_VERSION': '', 'TEST_MODE': mode,
                           'CALL_LOG': str(log), 'BOOTSTRAP_MARKER': str(root / 'bootstrapped')}
            result = subprocess.run(['bash', str(script)], env=environment, capture_output=True, text=True)
            return result.returncode, log.read_text().splitlines() if log.exists() else []

    def test_existing_data_skips_bootstrap(self):
        self.assertEqual(self.run_workflow('valid'), (0, ['validate', 'benchmark']))

    def test_missing_data_bootstraps_then_revalidates(self):
        self.assertEqual(self.run_workflow('missing'), (0, ['validate', 'bootstrap', 'validate', 'benchmark']))

    def test_failed_revalidation_prevents_benchmark(self):
        code, calls = self.run_workflow('broken')
        self.assertNotEqual(code, 0)
        self.assertEqual(calls, ['validate', 'bootstrap', 'validate'])

    def test_invalid_run_id_stops_before_data_access(self):
        self.assertEqual(self.run_workflow('valid', '../bad'), (2, []))


if __name__ == '__main__':
    unittest.main()
