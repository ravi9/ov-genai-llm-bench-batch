#!/bin/bash

python3 -m venv ov-genai-env
source ov-genai-env/bin/activate

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu  
python -m pip install "optimum-intel[openvino]"@git+https://github.com/huggingface/optimum-intel.git
pip install openvino-genai

git clone https://github.com/openvinotoolkit/openvino.genai.git

LLM_BENCH_REQ="openvino.genai/tools/llm_bench/requirements.txt"
if [ ! -f "$LLM_BENCH_REQ" ]; then
	echo "Missing LLM_BENCH requirements file: $LLM_BENCH_REQ" >&2
	exit 1
fi

# Create sanitized copy at project root (outside third-party directory)
sed '/--extra-index-url https:\/\/storage.openvinotoolkit.org\/simple\/wheels\/nightly/d' "$LLM_BENCH_REQ" > requirements-bench.txt
pip install -r requirements-bench.txt