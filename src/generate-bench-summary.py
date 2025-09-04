#!/usr/bin/env python3
"""Aggregate benchmark JSON files into a CSV summary.

Automatically invoked by run-llm-bench-batch.py after benchmark runs complete.

Filename convention:
  ${MODEL_NAME}#${WEIGHT_FORMAT}#${SYM_LABEL}#g_${GROUP_SIZE}#ov#${DEVICE}.json

python generate-bench-summary.py --reports-dir benchmark-reports-20250904-145700/

"""
import argparse, csv, json, os, sys
from typing import Dict, Any, List

EXPECTED_SEGMENTS = 6


def parse_filename(filename: str):
    name = filename[:-5] if filename.endswith('.json') else filename
    parts = name.split('#')
    if len(parts) != EXPECTED_SEGMENTS:
        raise ValueError('segment mismatch')
    model_name, weight_format, sym_label, group_part, framework, device = parts
    if not group_part.startswith('g_'):
        raise ValueError('group part malformed')
    return {
        'model_name': model_name,
        'weight_format': weight_format,
        'sym_label': sym_label,
        'group_size': group_part[2:],
        'framework': framework,
        'device': device,
    }


def extract_metrics(data: Dict[str, Any]):
    perf = data.get('perfdata') or {}
    results = perf.get('results') or []
    if not results:
        raise ValueError('missing results')
    first = results[0]
    avg = perf.get('results_averaged') or {}
    return {
        'input_size': first.get('input_size'),
        'infer_count': first.get('infer_count'),
        'generation_time': avg.get('generation_time'),
        'latency': avg.get('latency'),
        'first_latency': avg.get('first_latency'),
        'second_avg_latency': avg.get('second_avg_latency'),
        'first_infer_latency': avg.get('first_infer_latency'),
        'second_infer_avg_latency': avg.get('second_infer_avg_latency'),
        'tokenization_time': avg.get('tokenization_time'),
        'detokenization_time': avg.get('detokenization_time'),
        'compile_time_sec': perf.get('compile_time')
    }


COLUMNS = [
    'model_name', 'weight_format', 'sym_label', 'group_size', 'framework', 'device',
    'input_tokens', 'output_tokens', 'compile_time (ms)',
    'generation_time (ms)', 'latency (ms)', 'first_latency (ms)', 'second_avg_latency (ms)',
    'first_infer_latency (ms)', 'second_infer_avg_latency (ms)', 'second_infer_avg_throughput (t/s)',
    'tokenization_time (ms)', 'detokenization_time (ms)'
]


def aggregate(directory: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith('.json'):
            continue
        try:
            meta = parse_filename(fname)
        except Exception as e:
            print(f"[warn] skip (name) {fname}: {e}", file=sys.stderr)
            continue
        data_path = os.path.join(directory, fname)
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            metrics = extract_metrics(data)
        except Exception as e:
            print(f"[warn] skip (json) {fname}: {e}", file=sys.stderr)
            continue
        second_infer_avg_latency = metrics.get('second_infer_avg_latency')
        if second_infer_avg_latency and second_infer_avg_latency > 0:
            throughput = 1000.0 / second_infer_avg_latency
        else:
            throughput = None
        compile_time_sec = metrics.get('compile_time_sec')
        compile_time_ms = compile_time_sec * 1000.0 if isinstance(compile_time_sec, (int, float)) else None
        row = {
            'model_name': meta['model_name'],
            'weight_format': meta['weight_format'],
            'sym_label': meta['sym_label'],
            'group_size': meta['group_size'],
            'framework': meta['framework'],
            'device': meta['device'],
            'input_tokens': metrics.get('input_size'),
            'output_tokens': metrics.get('infer_count'),
            'compile_time (ms)': compile_time_ms,
            'generation_time (ms)': metrics.get('generation_time'),
            'latency (ms)': metrics.get('latency'),
            'first_latency (ms)': metrics.get('first_latency'),
            'second_avg_latency (ms)': metrics.get('second_avg_latency'),
            'first_infer_latency (ms)': metrics.get('first_infer_latency'),
            'second_infer_avg_latency (ms)': second_infer_avg_latency,
            'second_infer_avg_throughput (t/s)': throughput,
            'tokenization_time (ms)': metrics.get('tokenization_time'),
            'detokenization_time (ms)': metrics.get('detokenization_time'),
        }
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description='Generate benchmark CSV summary from JSON reports.')
    ap.add_argument('--reports-dir', required=True)
    ap.add_argument('--output', help='Defaults to <reports-dir>.csv')
    args = ap.parse_args()
    if not os.path.isdir(args.reports_dir):
        print(f"Reports directory not found: {args.reports_dir}", file=sys.stderr)
        return 2
    out = args.output
    if out is None:
        base = os.path.basename(os.path.normpath(args.reports_dir)) or 'benchmark_summary'
        out = base + '.csv'
    rows = aggregate(args.reports_dir)
    if not rows:
        print('No valid JSON reports found.', file=sys.stderr)
        return 1
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {len(rows)} row(s) -> {os.path.abspath(out)}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
