"""Shared preflight for the complete deployed detector precision set."""
from pathlib import Path
import argparse

PRECISIONS = ('fp32', 'fp16', 'int8')


def require_detector_artifacts(models_dir, *, read_models=False):
    directory = Path(models_dir)
    paths = {precision: directory / f'person_detector_{precision}.xml' for precision in PRECISIONS}
    missing = [str(path) for xml in paths.values() for path in (xml, xml.with_suffix('.bin'))
               if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError('Benchmark requires complete, nonempty FP32, FP16 and INT8 XML/BIN pairs. '
                           'Finish the production training/optimization pipeline first. Missing or empty: '
                           + ', '.join(missing))
    if read_models:
        import openvino as ov
        core = ov.Core()
        for precision, xml in paths.items():
            try:
                core.read_model(str(xml), str(xml.with_suffix('.bin')))
            except Exception as exc:
                raise RuntimeError(f'Invalid {precision} OpenVINO model pair: {xml}: {exc}') from exc
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models-dir', type=Path, required=True)
    parser.add_argument('--read-models', action='store_true', help='Also validate each pair with OpenVINO')
    args = parser.parse_args()
    require_detector_artifacts(args.models_dir, read_models=args.read_models)
    print(f'Complete FP32/FP16/INT8 model set: {args.models_dir}')


if __name__ == '__main__':
    main()
