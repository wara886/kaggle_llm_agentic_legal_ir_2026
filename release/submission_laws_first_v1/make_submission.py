from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path


def _to_cli_value(v):
    if isinstance(v, bool):
        return 'true' if v else 'false'
    return str(v)


def _load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def _build_cmd(root: Path, cfg: dict, use_train_aug: bool) -> tuple[list[str], Path, Path]:
    pipeline = root / cfg['pipeline_script']
    out_dir = root / cfg['runtime_output_dir']
    output_submission = root / cfg['output_submission_csv']

    flags = dict(cfg['base_flags'])
    if use_train_aug:
        aug = cfg.get('training_aug_option', {})
        if not aug.get('available', False):
            raise SystemExit('Training augmentation is not available in this repository state.')
        for k, v in aug.get('override_flags', {}).items():
            flags[k] = v

    cmd = [sys.executable, str(pipeline), '--out-dir', str(out_dir)]
    for k, v in flags.items():
        cmd.extend([f"--{k.replace('_', '-')}", _to_cli_value(v)])
    return cmd, out_dir, output_submission


def _copy_submission_from_runtime(root: Path, out_dir: Path, dst: Path) -> None:
    candidates = [
        out_dir / 'test_predictions_silver_baseline_v0.csv',
        root / 'submissions' / 'submission_silver_baseline_v0.csv',
    ]
    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        raise SystemExit('No submission artifact found after pipeline run.')

    dst.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with src.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({'query_id': row.get('query_id', ''), 'predicted_citations': row.get('predicted_citations', '')})

    with dst.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['query_id', 'predicted_citations'])
        writer.writeheader()
        writer.writerows(rows)



def main() -> None:
    parser = argparse.ArgumentParser(description='Submission wrapper for frozen laws-first v1 release package.')
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parent / 'submit_config.json')
    parser.add_argument('--enable-train-aug', action='store_true', help='Use optional validated training-aug model override.')
    args = parser.parse_args()

    cfg = _load_config(args.config)
    root = Path(__file__).resolve().parents[2]

    default_aug = bool(cfg.get('enable_training_aug_default', False))
    use_train_aug = bool(args.enable_train_aug or default_aug)

    cmd, out_dir, output_submission = _build_cmd(root, cfg, use_train_aug)

    print('[make_submission] Running frozen pipeline command:')
    print(' '.join(cmd))
    subprocess.run(cmd, check=True)

    _copy_submission_from_runtime(root, out_dir, output_submission)
    print(f'[make_submission] Submission written to: {output_submission}')


if __name__ == '__main__':
    main()
