# OpenVINO GenAI LLM Benchmark - Batch Workflow

This repository layout provides a reproducible workflow to:

1. Create an isolated Python environment with [OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai/tree/master/tools/llm_bench) + [Optimum-Intel](https://huggingface.co/docs/optimum/main/en/intel/installation)
2. Export Hugging Face LLMs to OpenVINO IR with quantization / group size variants
3. Smoke‑test exported model directories (simple text generation)
4. Generate fixed length prompt files per model for consistent benchmarking
5. Run multi‑model, multi‑device performance benchmarks (CPU / GPU / NPU)
6. Aggregate JSON benchmark outputs into a consolidated CSV report
7. (Optional) Fast subset iteration with `--limit-models` for quick validation

---
## Directory / File Overview

| File | Purpose |
|------|---------|
| `setup.sh` | Creates virtualenv, installs dependencies (PyTorch CPU wheels, optimum-intel, openvino-genai, llm_bench requirements) |
| `export-models.sh` | Batch exports HF models into OpenVINO IR (quantized variants) using `optimum-cli export openvino` |
| `create-prompts.py` | Builds a fixed length (`N` tokens) prompt JSONL inside each model folder |
| `run-llm-bench-batch.py` | Standalone batch benchmark driver (calls upstream benchmark.py) |
| `bench_aggregate.py` | Standalone aggregation of JSON results to CSV |
| `benchmark-reports-<timestamp>/` | Auto-created directory containing per‑model JSON (or CSV) benchmark outputs |

Exported model directory naming convention (produced by `export-models.sh`):

```
${MODEL_NAME}#${WEIGHT_FORMAT}#${SYM_LABEL}#g_${GROUP_SIZE}#ov
```

Benchmark output file naming convention (added device + extension):

```
${MODEL_NAME}#${WEIGHT_FORMAT}#${SYM_LABEL}#g_${GROUP_SIZE}#ov#${DEVICE}.json
```

`bench_aggregate.py` splits on `#` and emits columns:
`model_name, weight_format, sym_label, group_size, framework, device, input_size, infer_count, generation_time, latency, ...`

---

## 1. Environment Setup

Run once (adjust Python path if needed). 

```bash
chmod +x setup.sh
./setup.sh
```

This creates `ov-genai-env/` virtual environment and installs:
* PyTorch (CPU wheels by default, edit for GPU if required)
* `optimum-intel` (latest from GitHub)
* `openvino-genai`
* llm_bench dependencies (sanitized `requirements-bench.txt`)

Activate later with:

```bash
source ov-genai-env/bin/activate
```

---
## 2. Export Models

* Edit `export-models.sh` to add more MODEL_IDS / group sizes / weight formats.
* Exported models saved in `ov-models/` (change `OUT_DIR_ROOT` in the script if desired).

```bash
chmod +x export-models.sh
# Default exports to ov-models directory
./export-models.sh 2>&1 | tee export-models-log.log
# OR pass the export dir
./export-models.sh ov-models-test 2>&1 | tee export-models-log.log
```

---
## 3. Smoke Test Models

Use the batch runner in non-benchmark mode (omit `--benchmark`) for a quick generation sanity check over all model directories:

```bash
python run-llm-bench-batch.py --models-root ov-models \
  --devices CPU \
  --max-new-tokens 32 \
  --prompt "The goal of AI is "
```

This prints a truncated sample output per model (and per device if multiple specified). Failures (load/generation errors) are counted and summarized.

Speed up quick validations during development by limiting how many model directories are processed:

```bash
python run-llm-bench-batch.py --models-root ov-models \
  --devices CPU \
  --limit-models 2 \
  --prompt "The goal of AI is " --max-new-tokens 16
```

`--limit-models` (alias `-lm`) applies an alphabetical slice after discovering valid IR folders; useful for sanity checks before full runs.

---
## 4. Create Fixed-Length Prompts

Generates a JSONL like `prompt_64_tokens.jsonl` inside every valid model folder.

```bash
python create-prompts.py --models-root ov-models-test --prompt-length 64 --device CPU
```

Resulting file per model: `prompt_<N>_tokens.jsonl` with structure:

```jsonl
{"prompt": "<text truncated to N tokens>"}
```

If the base essay prompt is too short for `--prompt-length`, you'll get an error—pick a smaller value or extend `BASE_PROMPT`.

---
## 5. Run Benchmarks (Multi‑Model, Multi‑Device)

From repo root:

```bash
python run-llm-bench-batch.py \
  --models-root ov-models-test \
  --benchmark \
  -pf prompt_64_tokens.jsonl \
  --bench-iters 3 \
  --all-devices
```

Key options:
* `--benchmark` switch => use full `benchmark.py` instead of simple generation
* `-pf` prompt file basename (must exist inside each model dir)
* `--bench-iters` passed as `-n` iterations to underlying tool (first iteration may be warmup depending on config)
* Device control: `--devices CPU` (single), `--devices CPU,GPU` (list), or `--all-devices`
* Subset run: `--limit-models 3` (first 3 model dirs alphabetically)
* Report directory auto-created: `benchmark-reports-<timestamp>/`
* `--report-format json|csv` (default json)

Each successful run produces one JSON (or CSV) per model/device.

Example produced file:
```
Llama-3.2-1B-Instruct#int4#asym#g_128#ov#CPU.json
```

Contains fields:
* `perfdata.results[0]` (iteration 0 metrics; we take `input_size`, `infer_count`)
* `perfdata.results_averaged` (averaged latencies & timing fields for final row)

---
## 6. Aggregate Results to CSV

From repo root:

```bash
python bench_aggregate.py --reports-dir benchmark-reports-20250903-230104
```

Outputs (by default):
```
benchmark-reports-20250903-230104.csv
```

Columns:
```
model_name,weight_format,sym_label,group_size,framework,device,\
input_size,infer_count,generation_time,latency,first_latency,second_avg_latency,\
first_infer_latency,second_infer_avg_latency,tokenization_time,detokenization_time
```

Warnings are printed for malformed filenames or JSON structures and those files are skipped.

---
## 7. Typical End-to-End Session

```bash
source ov-genai-env/bin/activate
./export-models.sh
# Smoke test (no benchmark)
python run-llm-bench-batch.py --models-root ov-models-test --devices CPU --prompt "The goal of AI is " --max-new-tokens 32
# Create fixed-length prompts for benchmarking
python create-prompts.py --models-root ov-models-test --prompt-length 64 --device CPU
# Full benchmark
python run-llm-bench-batch.py --models-root ov-models-test --benchmark -pf prompt_64_tokens.jsonl --bench-iters 3 --all-devices
# Aggregate results. Replace `<timestamp>` with the actual directory printed during benchmarking.
python bench_aggregate.py --reports-dir benchmark-reports-<timestamp>
```

---
## 8. Customization Tips

* Add more models: edit `MODEL_IDS` in `export-models.sh`.
* Add more quantization/group size combos: extend `ASYM_GROUP_SIZES` / `SYM_GROUP_SIZES` arrays.
* Change report format: `--report-format csv`.
* Single device only: omit `--all-devices` and use `--devices GPU`.
* Skip prompt permutation control: add `--keep-prompt-permutation`.
* Faster iteration: `--limit-models N` to exercise only the first N model directories (alphabetical by name).

---
## 9. Troubleshooting

| Issue | Cause / Fix |
|-------|-------------|
| `Models root not found` | Path typo; verify with `ls` and adjust `--models-root` |
| Missing prompt file warning | Run `create-prompts.py` or ensure correct `-pf` basename |
| Negative latency values in JSON | Upstream tool anomaly / clock source; verify environment & rerun |
| No GPU / NPU runs | Ensure drivers + OpenVINO plugin installed and visible (check `benchmark.py -h`) |

---
## 10. Quick Subset Runs (`--limit-models`)

When prototyping changes (e.g., adjusting prompt length, verifying new export parameters), running the entire model matrix can be slow. Use:

```bash
python run-llm-bench-batch.py --models-root ov-models-test --benchmark -pf prompt_64_tokens.jsonl \
  --bench-iters 2 --devices CPU,GPU --limit-models 1
```

Notes:
* Ordering is deterministic (sorted directory names); to test a specific model ensure its name sorts early or temporarily move it to a separate root.
* The reported summary still reflects only the processed subset.
* Combine with fewer `--bench-iters` for even faster smoke performance validation of the benchmark pipeline.

## 11. Next Ideas

* Add Markdown summary generation from CSV
* Track git commit + environment metadata inside each JSON / CSV row
* Add charts (latency vs group size)

---
