#!/usr/bin/env python3
"""
Runs lightweight generation smoke test OR full benchmark
for every valid exported OpenVINO GenAI model directory under --models-root.

Prompt file (-pf) is expected inside each model directory when --benchmark is used.

Benchmark report files are written under a timestamped directory unless --reports-root provided.

time python run-llm-bench-batch.py  \
    --models-root ov-models-test-1  \
    --benchmark  \
    -pf prompt_64_tokens.jsonl \
    --bench-iters 3 \
    --all-devices
"""

import argparse
import os
import sys
import subprocess
from datetime import datetime
import openvino_genai  # type: ignore


def is_model_dir(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    return any(f.endswith('.xml') for f in os.listdir(path))


def run_model(model_dir: str, prompt: str, max_new_tokens: int, device: str) -> str:
    pipe = openvino_genai.LLMPipeline(model_dir, device)
    config = openvino_genai.GenerationConfig()
    config.max_new_tokens = max_new_tokens
    config.temperature = 0.0
    return pipe.generate(prompt, config)


def run_benchmark(model_dir: str, prompt_file: str, num_iters: int, device: str, reports_root: str,
                  disable_prompt_permutation: bool = True, report_format: str = 'json') -> tuple[int, str]:
    bench_script = os.path.join(os.path.dirname(__file__), 'openvino.genai', 'tools', 'llm_bench', 'benchmark.py')
    if not os.path.isfile(bench_script):
        print(f"benchmark.py not found at {bench_script}", file=sys.stderr)
        return 127
    os.makedirs(reports_root, exist_ok=True)
    model_name = os.path.basename(os.path.normpath(model_dir))
    ext = 'json' if report_format == 'json' else 'csv'
    report_path = os.path.join(reports_root, f"{model_name}#{device.upper()}.{ext}")
    report_flag = ['-rj', report_path] if report_format == 'json' else ['-r', report_path]
    cmd = [
        sys.executable, bench_script,
        '-m', model_dir,
        '-pf', prompt_file,
        '-n', str(num_iters),
        '-d', device,
    ] + report_flag
    if disable_prompt_permutation:
        cmd.append('--disable_prompt_permutation')
    print(f"[benchmark] Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        print(f"[benchmark] FAILED ({proc.returncode}) for {model_name}", file=sys.stderr)
    else:
        print(f"[benchmark] OK -> {report_path}")
    return proc.returncode, proc.stdout


def main():
    parser = argparse.ArgumentParser(description="Standalone OpenVINO GenAI multi-model test / benchmark runner.")
    parser.add_argument('--models-root', '-mr', default='ov-models-test', help='Root containing exported model subdirectories.')
    parser.add_argument('--limit-models', '-lm', type=int, help='Number of models to limit processing. For testing.')
    parser.add_argument('--prompt-file', '-pf', default='prompt.jsonl', help='Prompt filename expected inside each model dir (benchmark mode).')
    parser.add_argument('--prompt', '-p', default='What is artificial intelligence?', help='Prompt for smoke test mode.')
    parser.add_argument('--max-new-tokens', '-mnt', type=int, default=64, help='Max new tokens for smoke test.')
    parser.add_argument('--devices', '-ds', default='CPU', help='Device list (single or comma-separated, e.g. CPU or CPU,GPU,NPU). Default: CPU')
    parser.add_argument('--all-devices', '-ad', action='store_true', help='Shortcut for CPU,GPU,NPU.')
    parser.add_argument('--benchmark', '-b', action='store_true', help='Enable full benchmark mode.')
    parser.add_argument('--bench-iters', '-bi', type=int, default=3, help='Iterations (-n) for benchmark.py.')
    parser.add_argument('--reports-root', '-rr', help='Directory for benchmark outputs (default: benchmark-reports-<timestamp>).')
    parser.add_argument('--report-format', '-rf', choices=['json', 'csv'], default='json', help='Benchmark report format.')
    parser.add_argument('--keep-prompt-permutation', '-kpp', action='store_true', help='Keep prompt permutation (default disables).')
    args = parser.parse_args()

    if not os.path.isdir(args.models_root):
        print(f"Models root not found: {args.models_root}", file=sys.stderr)
        return 2

    model_dirs = [os.path.join(args.models_root, d) for d in sorted(os.listdir(args.models_root))]
    model_dirs = [d for d in model_dirs if is_model_dir(d)]
    if args.limit_models:
        model_dirs = model_dirs[:args.limit_models]

    if not model_dirs:
        print('No model directories with IR xml found.', file=sys.stderr)
        return 1

    if args.all_devices:
        devices = ['CPU', 'GPU', 'NPU']
    else:
        devices = [d.strip() for d in args.devices.split(',') if d.strip()]
    # Deduplicate & normalize case
    norm = []
    seen = set()
    for dev in devices:
        u = dev.upper()
        if u and u not in seen:
            seen.add(u)
            norm.append(u)
    devices = norm

    print(f"Found {len(model_dirs)} model(s). Devices: {', '.join(devices)}\n")
    failures = 0
    prompt_basename = os.path.basename(args.prompt_file)

    if args.benchmark:
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        reports_root = args.reports_root or f"benchmark-reports-{timestamp}"
        print(f"Benchmark reports dir: {reports_root}")
        error_log_path = f"benchmark-errors-{timestamp}.log"
        print(f"Failure log (if any): {error_log_path}")
        for d in model_dirs:
            name = os.path.basename(d)
            # Extract sym/asym label from directory name: MODEL#WEIGHT#SYM_LABEL#g_GROUP#ov
            parts = name.split('#')
            sym_label = parts[2] if len(parts) >= 3 else ''
            for dev in devices:
                if dev.upper() == 'NPU' and sym_label.lower() == 'asym':
                    print(f"Skipping {name} [NPU] - asym quantization unsupported on NPU")
                    continue
                model_prompt_file = os.path.join(d, prompt_basename)
                if not os.path.isfile(model_prompt_file):
                    print(f"Skipping {name} [{dev}] - missing prompt file {model_prompt_file}", file=sys.stderr)
                    failures += 1
                    continue
                print(f"=== BENCH {name} [{dev}] ===")
                ret, out_text = run_benchmark(
                    model_dir=d,
                    prompt_file=model_prompt_file,
                    num_iters=args.bench_iters,
                    device=dev,
                    reports_root=reports_root,
                    disable_prompt_permutation=not args.keep_prompt_permutation,
                    report_format=args.report_format,
                )
                if ret != 0:
                    failures += 1
                    try:
                        with open(error_log_path, 'a', encoding='utf-8') as ef:
                            ef.write(f"===== FAILURE: model={name} device={dev} time={datetime.now().isoformat()} =====\n")
                            ef.write(out_text)
                            if not out_text.endswith('\n'):
                                ef.write('\n')
                    except Exception as log_e:
                        print(f"[warn] Could not write failure log: {log_e}", file=sys.stderr)
    else:
        for d in model_dirs:
            name = os.path.basename(d)
            parts = name.split('#')
            sym_label = parts[2] if len(parts) >= 3 else ''
            for dev in devices:
                if dev.upper() == 'NPU' and sym_label.lower() == 'asym':
                    print(f"Skipping {name} [NPU] - asym quantization unsupported on NPU")
                    continue
                print(f"=== {name} [{dev}] ===")
                try:
                    out = run_model(d, args.prompt, args.max_new_tokens, dev)
                    snippet = out.strip().replace('\n', ' ') + '\n'
                    print(snippet)
                except Exception as e:
                    failures += 1
                    print(f"ERROR: {e}\n", file=sys.stderr)

    total_executions = len(model_dirs) * len(devices)
    print(f"Completed. Failures: {failures} / {total_executions} (models * devices = {len(model_dirs)} * {len(devices)})")
    return 0 if failures == 0 else 1


if __name__ == '__main__':
    sys.exit(main())

