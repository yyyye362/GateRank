from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
from typing import List, Union
import itertools

import numpy as np
import tqdm
import fire

from eval_lcd.data import read_jsonl, write_jsonl, read_problems, get_problem_file, get_nested, code_extract
from eval_lcd.execution import check_correctness


def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int
) -> np.ndarray:
    """Estimates pass@k of each problem and returns them in an array."""
    def estimator(n: int, c: int, k: int) -> float:
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])


def evaluate_functional_correctness(
    sample_file: str,
    problem_file: str,
    k: List[int],
    n_workers: int = 4,
    timeout: float = 3.0,
):
    problems = read_problems(problem_file)

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        completion_id = Counter()
        n_samples = 0
        results = {}
        pass_count = defaultdict(int)

        print("Reading samples...")
        for idx, sample in enumerate(tqdm.tqdm(read_jsonl(sample_file))):
            task_id = sample["task_id"]
            completion = sample["completion"]
            args = (problems[task_id], completion, timeout, completion_id[task_id])
            future = executor.submit(check_correctness, *args)
            futures.append((idx, task_id, completion_id[task_id], future))
            completion_id[task_id] += 1
            n_samples += 1

        assert len(completion_id) == len(problems), "Some problems are not attempted."

        print("Running test suites...")
        for idx, task_id, comp_id, future in tqdm.tqdm(futures, total=len(futures)):
            result = future.result()
            results[idx] = {
                "task_id": task_id,
                "completion_id": comp_id,
                "result": result["result"],
                "passed": result["passed"],
            }
            if result["passed"]:
                pass_count[task_id] += 1

    # 计算 pass@k
    task_to_pass = defaultdict(list)
    for r in results.values():
        task_to_pass[r["task_id"]].append(r["passed"])

    total = np.array([len(v) for v in task_to_pass.values()])
    correct = np.array([sum(v) for v in task_to_pass.values()])

    ks = k
    pass_at_k = {
        f"pass@{k}": estimate_pass_at_k(total, correct, k).mean()
        for k in ks if (total >= k).all()
    }

    def combine_results():
        for idx, sample in enumerate(read_jsonl(sample_file)):
            res = results[idx]
            ordered = {
                "task_id": res["task_id"],
                "completion_id": res["completion_id"],
                "passed": res["passed"],
                "result": res["result"],
            }
            for k, v in sample.items():
                if k not in ordered:
                    ordered[k] = v
            yield ordered


    out_file = sample_file.replace('.jsonl', '_results.jsonl')
    print(f"Writing results to {out_file}...")
    write_jsonl(out_file, tqdm.tqdm(combine_results(), total=n_samples))


    summary_lines = ["=== 通过题目统计 ==="]
    for task_id, cnt in sorted(pass_count.items(), key=lambda x: (-x[1], x[0])):
        summary_lines.append(f"{task_id}  {cnt}")

    summary_lines.append("\n=== pass@k 结果 ===")
    for k_, v in pass_at_k.items():
        summary_lines.append(f"{k_}: {v:.4f}")

    summary_file = sample_file.replace('.jsonl', '_results.txt')
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    print(f"\nSummary saved to {summary_file}")

    return pass_at_k


def evaluate(
        input_file: str, 
        predict_column: str = 'completion',
        version: str = 'v0.3.1',
        split: str = 'test',
        k: str = "1,5,10,20,100"):
    problem_file = get_problem_file(version, split)
    k = list(map(int, k.split(",")))


    result = []
    for sample in read_jsonl(input_file):
        assert 'task_id' in sample, f'`task_id` should be specified in {input_file}'
        text = get_nested(sample, predict_column)
        sample['completion'] = code_extract(text)
        result.append(sample)
    assert '.jsonl' in input_file, 'input_file must be a jsonl file'
    sample_file = input_file.replace('.jsonl', '_sample.jsonl')
    write_jsonl(sample_file, result)

    results = evaluate_functional_correctness(sample_file, problem_file, k)
    print(results)


def cli():
    fire.Fire(evaluate)


if __name__ == '__main__':
    cli()
