from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path


def _load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def _split_citations(text: str) -> list[str]:
    if text is None:
        return []
    return [x.strip() for x in str(text).split(';') if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description='Preflight checks for frozen submission package.')
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parent / 'submit_config.json')
    args = parser.parse_args()

    cfg = _load_config(args.config)
    root = Path(__file__).resolve().parents[2]
    test_path = root / cfg['input_test_csv']
    sub_path = root / cfg['output_submission_csv']

    if not test_path.exists():
        raise SystemExit(f'test.csv not found: {test_path}')
    if not sub_path.exists():
        raise SystemExit(f'submission.csv not found: {sub_path}')

    with test_path.open('r', encoding='utf-8-sig', newline='') as f:
        test_rows = list(csv.DictReader(f))
    with sub_path.open('r', encoding='utf-8-sig', newline='') as f:
        sub_reader = csv.DictReader(f)
        sub_rows = list(sub_reader)
        fieldnames = sub_reader.fieldnames or []

    expected_cols = ['query_id', 'predicted_citations']
    if fieldnames != expected_cols:
        raise SystemExit(f'Invalid submission columns: {fieldnames}, expected: {expected_cols}')

    test_ids = [r.get('query_id', '') for r in test_rows]
    sub_ids = [r.get('query_id', '') for r in sub_rows]

    if len(sub_rows) != len(test_rows):
        raise SystemExit(f'Row count mismatch: submission={len(sub_rows)} vs test={len(test_rows)}')

    missing_ids = sorted(set(test_ids) - set(sub_ids))
    if missing_ids:
        raise SystemExit(f'Missing query_id in submission: {missing_ids[:10]}')

    non_str_rows = []
    duplicate_rows = []
    for idx, row in enumerate(sub_rows, start=1):
        pc = row.get('predicted_citations', '')
        if not isinstance(pc, str):
            non_str_rows.append(idx)
            continue
        citations = _split_citations(pc)
        if len(citations) != len(set(citations)):
            duplicate_rows.append((idx, row.get('query_id', '')))

    if non_str_rows:
        raise SystemExit(f'predicted_citations has non-string rows: {non_str_rows[:10]}')
    if duplicate_rows:
        raise SystemExit(f'Duplicate citations detected in rows: {duplicate_rows[:10]}')

    # writable csv check
    tmp_dir = sub_path.parent
    with tempfile.NamedTemporaryFile('w', delete=False, dir=tmp_dir, suffix='.csv', encoding='utf-8-sig', newline='') as tmp:
        writer = csv.DictWriter(tmp, fieldnames=expected_cols)
        writer.writeheader()
        for row in sub_rows:
            writer.writerow({'query_id': row.get('query_id', ''), 'predicted_citations': row.get('predicted_citations', '')})
        tmp_path = Path(tmp.name)
    tmp_path.unlink(missing_ok=True)

    print('[preflight_check] PASS')
    print(f'[preflight_check] test rows: {len(test_rows)}')
    print(f'[preflight_check] submission rows: {len(sub_rows)}')
    print(f'[preflight_check] checked file: {sub_path}')


if __name__ == '__main__':
    main()
