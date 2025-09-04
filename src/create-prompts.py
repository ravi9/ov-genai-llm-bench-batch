"""
Generate a fixed-length prompt file in each model directory under a root.
python create-prompts.py --models-root ov-models --prompt-length 64 --device CPU
"""
import argparse
import os
import sys
import json
import openvino as ov
import openvino_genai as ov_genai


BASE_PROMPT = """
Write a detailed and comprehensive essay on the history and development of artificial intelligence, 
covering its origins in philosophy, logic, and early computing. Begin with the foundational ideas 
in the works of Aristotle, Leibniz, and Boole, and continue through the invention of the Turing machine 
and the formalization of computation. Discuss the birth of AI as a field at the 1956 Dartmouth Conference, 
and the early optimism around symbolic AI, logic-based systems, and problem-solving programs like the Logic Theorist 
and General Problem Solver. Describe the rise and fall of expert systems in the 1970s and 1980s, 
including their commercial success and eventual limitations. Cover the 'AI winters' and the criticism 
from scientists like Hubert Dreyfus and Marvin Minsky. Transition into the resurgence of AI through 
machine learning, especially neural networks, backpropagation, and the availability of large datasets 
and computational power. Explain the breakthroughs in deep learning: AlexNet in 2012, the rise of CNNs 
for computer vision, RNNs and LSTMs for sequence modeling, and the transformer architecture introduced 
in 'Attention Is All You Need' in 2017. Discuss how transformers enabled large language models like BERT, 
GPT, and eventually Llama, leading to the current era of foundation models. Cover key concepts such as 
self-attention, pretraining and fine-tuning, prompt engineering, and in-context learning. 
Explain the scaling laws that govern model performance with respect to model size, dataset size, 
and compute. Discuss the societal impact of AI: automation, job displacement, bias and fairness, 
misinformation, and existential risk. Cover ethical frameworks, alignment research, and governance 
efforts. Include recent advances such as multimodal models (e.g., CLIP, Flamingo), agent-based systems, 
and AI in science (e.g., AlphaFold). Conclude with open challenges: reasoning, robustness, interpretability, 
energy efficiency, and the path toward artificial general intelligence. Ensure the essay is well-structured, 
logically organized, and written in clear, academic prose. Use examples and historical milestones to illustrate 
key points. Avoid overly technical jargon where possible, but do not oversimplify core concepts. 
Include discussions of both technical and philosophical dimensions of AI. Address debates such as 
symbolic vs. connectionist AI, the Chinese Room argument, and the nature of consciousness. 
Also discuss the role of major institutions (e.g., DeepMind, OpenAI, FAIR), open-source movements, 
and geopolitical competition in shaping the field. Finally, reflect on the future trajectory of AI 
and its potential to transform science, medicine, education, and society as a whole.
"""


def discover_model_dirs(root: str) -> list:
    if not os.path.isdir(root):
        return []
    out = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        # Require tokenizer & detokenizer IRs
        tok = os.path.join(path, "openvino_tokenizer.xml")
        detok = os.path.join(path, "openvino_detokenizer.xml")
        if os.path.isfile(tok) and os.path.isfile(detok):
            out.append(path)
    return out

def build_prompt_for_model(core: ov.Core, model_dir: str, prompt_len: int, device: str) -> tuple[str, int]:  
    
    try:  
        pipe = ov_genai.LLMPipeline(model_dir, device)  
    except Exception as e:  
        raise RuntimeError(f"Failed to load LLMPipeline from {model_dir}. Make sure it's GenAI-exported. Error: {e}")  
  
    tokenizer = pipe.get_tokenizer()
    tokens = tokenizer.encode(BASE_PROMPT)
    seq_len = tokens.input_ids.shape[-1]

    if seq_len < prompt_len:
        raise ValueError(f"Base prompt is too short: {seq_len} tokens (< {prompt_len}). Use a longer prompt.")

    token_data = tokens.input_ids.data
    truncated_data = token_data[0, :prompt_len]
    truncated_list = truncated_data.tolist()

    final_prompt = tokenizer.decode(truncated_list)

    # Verify the length  
    verified_tokens = tokenizer.encode(final_prompt)  
    verified_len = verified_tokens.input_ids.shape[-1]  
  
    if verified_len != prompt_len:  
        raise RuntimeError(f"Length mismatch after round-trip: {verified_len} != {prompt_len}")  
      
    return final_prompt, verified_len


def main():
    parser = argparse.ArgumentParser(description="Generate a fixed-length prompt file in each model directory.")
    parser.add_argument('--models-root', '-m', default='ov-models-test', help='Root directory containing model folders.')
    parser.add_argument('--prompt-length', '-l', type=int, default=64, help='Target token length.')
    parser.add_argument('--device', '-d', default='CPU', help='Device for tokenizer/detokenizer.')
    parser.add_argument('--output-basename', '-o', default='prompt', help='Prefix for output file name. <basename>_<N>_tokens.jsonl')

    args = parser.parse_args()

    models_root = os.path.abspath(args.models_root)
    model_dirs = discover_model_dirs(models_root)
    
    if not model_dirs:
        print(f"No model directories with tokenizer+detokenizer found under: {models_root}", file=sys.stderr)
        return 1

    print(f"Found {len(model_dirs)} model directories. Target length: {args.prompt_length}\n")
    core = ov.Core()
    successes = 0
    failures = 0
    for m_dir in model_dirs:
        name = os.path.basename(m_dir)
        out_path = os.path.join(m_dir, f"{args.output_basename}_{args.prompt_length}_tokens.jsonl")
        try:
            prompt, length = build_prompt_for_model(core, m_dir, args.prompt_length, args.device)
            prompt_data = {"prompt": prompt}  
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(json.dumps(prompt_data) + '\n')
            print(f"[ok] {name}: wrote {length} tokens -> {os.path.basename(out_path)}")
            successes += 1
        except Exception as e:
            print(f"[error] {name}: {e}", file=sys.stderr)
            failures += 1

    print(f"\nDone. Success: {successes}  Failures: {failures}")
    return 0 if failures == 0 else 2


if __name__ == '__main__':
    sys.exit(main())