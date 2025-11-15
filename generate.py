from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import json
import glob
from tqdm import tqdm
import pprint
from peft import PeftModel
import sys
sys.set_int_max_str_digits(0) 

def generate_prompt(args, test_case_path, prompt_path, solutions_path, tokenizer, starter_path=None):
    _input = "\nQUESTION:\n"
    with open(prompt_path, "r") as f:
        data = f.read()
    _input += data

    if starter_path is not None:
        with open(starter_path, "r") as f:
            data = f.read()
        _input += "\n" + data

    if os.path.exists(test_case_path):
        with open(test_case_path, "r") as f:
            data = json.load(f)
        if not data.get("fn_name"):
            _input += "\nUse Standard Input format"
        else:
            _input += "\nUse Call-Based format"
    elif starter_path is not None:
        _input += "\nUse Call-Based format"
    else:
        _input += "\nUse Standard Input format"

    _input += "\nANSWER:\n"
    return _input


def main(args):
    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    problems = sorted(glob.glob(args.test_path + '/*'))

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)

    print("Saving results to {}".format(args.output_path))

    start = max(0, args.start)
    end = min(args.end or len(problems), len(problems))
    problems = problems[start:end]

    base_model_name = "/home/yzj/model/Qwen2.5-Coder-7B-Instruct"  # 或者你训练时用的那个 HF 名称
    print(f"Loading base model `{base_model_name}` …")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name, 
        use_fast=False   # CodeLLaMA 建议关闭 fast tokenizer
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    print(f"Applying LoRA weights from `{args.model_path}` …")
    model = PeftModel.from_pretrained(
        base_model,
        args.model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    # model = AutoModelForCausalLM.from_pretrained(
    #     base_model_name,
    #     torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    #     device_map="auto"
    # )

    model.eval()

    for index, problem in tqdm(enumerate(problems), total=len(problems), ncols=0):
        prob_path = os.path.join(problem)
        problem_id = int(os.path.basename(problem))

        test_case_path = os.path.join(prob_path, "input_output.json")
        prompt_path = os.path.join(prob_path, "question.txt")
        starter_path = os.path.join(prob_path, "starter_code.py")
        solutions_path = os.path.join(prob_path, "solutions.json")

        if os.path.exists(os.path.join(args.output_path, f"{problem_id}.json")):
            continue

        if not os.path.exists(starter_path):
            starter_path = None

        input_text = generate_prompt(args, test_case_path, prompt_path, solutions_path, tokenizer, starter_path)
        
        input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=args.source_len).to(model.device)

        output_programs = []
        num_loops = int(args.num_seqs / args.num_seqs_per_iter)
        for _ in tqdm(range(num_loops), total=num_loops, leave=False, ncols=0):
            outputs = model.generate(
                input_ids,
                do_sample=True,
                temperature=args.temperature,
                max_length=args.max_len,
                num_return_sequences=args.num_seqs_per_iter,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id
            )
            for o in outputs:
                decoded = tokenizer.decode(o[input_ids.shape[-1]:], skip_special_tokens=True)
                output_programs.append(decoded)

        save_codes = {
            f"{problem_id}": {'codes': output_programs}
        }
        with open(os.path.join(args.output_path, f"{problem_id}.json"), 'w') as f:
            json.dump(save_codes, f)


if __name__ == "__main__":
    from Configs.generate_configs import *
    main(args)
