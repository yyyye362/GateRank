import json
import re
import ast
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


base_model_path = "Qwen2.5-Coder"
test_data_path = "LeetCodeDataset-v0.3.1-test.jsonl"
output_file_path = " "


def extract_completion(text: str) -> str:
    pattern = r"```(?:python)?\n(.*?)```"
    code_blocks = re.findall(pattern, text, re.S)
    if code_blocks:
        code_blocks.sort(key=lambda x: len(x.split('\n')))
        return code_blocks[-1].strip()
    return text.strip()

def syntax_check(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except (SyntaxError, MemoryError):
        return False

def code_extract(text: str) -> str:
    lines = text.split("\n")
    longest_line_pair = (0, 0)
    longest_so_far = 0
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            fragment = "\n".join(lines[i:j+1])
            if syntax_check(fragment):
                count = sum(1 for l in lines[i:j+1] if l.strip())
                if count > longest_so_far:
                    longest_so_far = count
                    longest_line_pair = (i, j)
    return "\n".join(lines[longest_line_pair[0]: longest_line_pair[1]+1])

def load_test_data(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_results(results, path):
    with open(path, 'w') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


print("Loading base model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()
print("Model ready for inference")


def generate_code(prompt, max_new_tokens=512):
    inputs = tokenizer(prompt, return_tensors='pt', padding=True,
                        truncation=True, max_length=1024).to(model.device)
    config = dict(
        max_new_tokens=max_new_tokens,
        temperature=0.3,
        top_p=0.95,
        top_k=50,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.1
    )
    with torch.no_grad():
        out = model.generate(input_ids=inputs.input_ids,
                             attention_mask=inputs.attention_mask,
                             **config)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    body = text[len(prompt):].strip()
    code = extract_completion(body)
    if not syntax_check(code):
        code = code_extract(body)
    return code


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=228)
    parser.add_argument('--num_generations', type=int, default=10)
    args = parser.parse_args()

    test_data = load_test_data(test_data_path)
    if args.num_samples:
        test_data = test_data[:args.num_samples]

    samples = []
    for sample in tqdm(test_data, desc='Gen'):
        tid = str(sample.get('task_id') or sample.get('id'))
        if not tid:
            raise KeyError("缺少 task_id 或 id，无法对齐问题")
        prompt = sample.get('query') or sample.get('prompt')
        for _ in range(args.num_generations):
            sol = generate_code(prompt)
            samples.append({
                'task_id': tid,
                'completion': sol
            })

    save_results(samples, output_file_path)
    print(f"Saved {len(samples)} samples to {output_file_path}")
