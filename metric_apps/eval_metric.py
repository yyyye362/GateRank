import numpy as np
import itertools
import os
from typing import Dict, Any, List
from result_style import deal_result

def estimate_pass_at_k(num_samples, num_correct, k):
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


def get_results(results: Dict[int, List[List[int]]], 
                count_errors: bool = False, 
                k_list: List[int] = [1, 10, 100]) -> Dict[str, Any]:
    """
    Computes metrics and returns a dict including:
      - avg_accuracy, strict_accuracy, pass_at_k
      - passed_ids: list of problem indices that passed at least once
      - passed_count: total number of problems that passed
    """
    metrics: Dict[str, Any] = {
        "avg_accuracy": None,
        "strict_accuracy": None,
        "pass_at_k": None,
        "passed_ids": [],
        "passed_count": 0
    }

    # single-generation case
    if len(results.get(0, [])) == 1:
        print("Computing accuracy metrics...")
        per_prob_res = []
        strict_pass_ids: List[int] = []
        for idx, gen_list in results.items():
            arr = np.array(gen_list[0])
            mean_pass = np.mean(arr > 0)
            if mean_pass == 1.0:
                strict_pass_ids.append(idx)
            per_prob_res.append(mean_pass)
        metrics["avg_accuracy"] = float(np.mean(per_prob_res))
        metrics["strict_accuracy"] = len(strict_pass_ids) / len(results)
        metrics["passed_ids"] = strict_pass_ids
        metrics["passed_count"] = len(strict_pass_ids)  # ✅ 添加通过数量

    else:
        print("Computing pass@k metric for multiple generations...")
        total = []
        correct = []
        passed_once: List[int] = []
        for idx, gens in results.items():
            # each gen passes if all test results >0
            flags = [all(np.array(g) > 0) for g in gens]
            total.append(len(flags))
            correct.append(sum(flags))
            if any(flags):
                passed_once.append(idx)
        ks = k_list
        total_arr = np.array(total)
        correct_arr = np.array(correct)
        pass_at_k = {
            f"pass@{k}": estimate_pass_at_k(total_arr, correct_arr, k).mean()
            for k in ks if (total_arr >= k).all()
        }
        metrics["pass_at_k"] = pass_at_k
        metrics["passed_ids"] = passed_once
        metrics["passed_count"] = len(passed_once)  # ✅ 添加通过数量
        print(pass_at_k)

    return metrics



def compute_metrics(results: Dict[int, List[List[int]]],
                    k_list: List[int] = [1, 2, 3, 5, 10],
                    count_errors: bool = True) -> Dict[str, Any]:
    return get_results(results, count_errors=count_errors, k_list=k_list)


if __name__ == "__main__":
    results = deal_result("/home/yzj/CodeRL/outputs1/3B_epoch5_sas1_results")
    # filter first 5000
    com_result = {i: results[i] for i in range(5000) if i in results}
    metrics = compute_metrics(com_result, [1, 2, 3, 5, 10], True)
    print("Metrics:", metrics)
    #print("Passed problem IDs:", metrics.get("passed_ids", []))
