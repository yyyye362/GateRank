import json
import numpy as np
import os
import glob 
from tqdm import tqdm
import pickle as pkl 
import multiprocessing
from testing_util import run_test

TIMEOUT = 4

def check_correctness(prob_path, generation, timeout, debug , example_tests):
    """Check correctness of code generation with a global timeout."""
    def _temp_run(prob_path, generation, debug, example_tests, result_result, result_error):
        tmp = run_test(prob_path=prob_path, test=generation, debug=debug, example_tests=example_tests)
        result_result.extend(tmp[0])
        result_error.extend(tmp[1])

    manager = multiprocessing.Manager()
    result_result = manager.list()
    result_error = manager.list()

    p = multiprocessing.Process(target=_temp_run, args=(prob_path, generation, debug, example_tests, result_result, result_error))
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.kill()
    if not result_result:
        result_result = [-1]
        result_error = [None]
        if debug:
            print(f"global timeout")
        return result_result, result_error
    return result_result, result_error

def eval_and_save_problems(args):
    problems = sorted(glob.glob(args.test_path + '/*'))
    test_indices = [] 
    for problem_idx, problem in enumerate(problems): 
        problem_id = int(problem.split('/')[-1])
        code_file_path = os.path.join(args.code_path, f'{problem_id}.json')
        if os.path.exists(code_file_path):
            test_indices.append(problem_idx)

    real_index = test_indices[args.index]
    problem = problems[real_index]

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    print(f'Testing sample {problem}')

    if args.example_tests:
        print("Using example tests")

    codes_loc = os.path.join(args.code_path, f'{real_index}.json')
    if not os.path.isfile(codes_loc):
        exit()
    with open(codes_loc, "r") as file: 
        gen_codes = json.load(file)[str(real_index)]['codes']

    test_file = os.path.join(problem, "input_output.json")
    if not os.path.isfile(test_file):
        exit()
    tests = json.load(open(test_file, 'r'))
    nb_tests = len(tests['inputs'])
    if args.max_tests != -1 and nb_tests > args.max_tests:
        exit()

    pkl_path = os.path.join(args.output_path, f'{real_index}.pkl')
    if os.path.isfile(pkl_path):
        print(f"remove file {pkl_path}")
        os.remove(pkl_path)

    print(f"Saving to {pkl_path}")

    all_results = []
    all_pass_counts = []

    for o_idx, o in tqdm(enumerate(gen_codes), total=len(gen_codes), ncols=0, leave=False):
        if args.debug:
            print(f"\n候选序号: {o_idx}\n{'='*80}\n")

        curr_results = []
        try:
            curr_results, _ = check_correctness(
                prob_path=problem,
                generation=o,
                timeout=TIMEOUT,
                debug=args.debug,
                example_tests=args.example_tests
            )

            fixed = []
            for e in curr_results:
                if isinstance(e, np.ndarray):
                    e = -2 if len(e) == 0 else e.item(0)
                if isinstance(e, np.bool_):
                    e = bool(e)
                fixed.append(e)
            curr_results = fixed

            pass_count = sum(1 for r in curr_results if r is True)
            all_pass_counts.append(pass_count)

        except Exception as e:
            print(f"test framework exception = {repr(e)}{e}\n")
            all_pass_counts.append(0)
            break
        finally:
            assert isinstance(curr_results, list)
            all_results.append(curr_results)

    save_results = {
        real_index: {
            'results': all_results,
            'pass_counts': all_pass_counts
        }
    }
    pkl.dump(save_results, open(pkl_path, "wb"))


    detailed_output = []
    for idx, res in enumerate(all_results):
        detailed_output.append({
            "candidate_id": idx,
            "pass_count": all_pass_counts[idx],
            "results": res
        })




    simple_json = {
        "pass_counts": all_pass_counts,
        "pass_bools": [p > 0 for p in all_pass_counts]
    }

    simple_json_path = os.path.join(args.output_path, f"{real_index}_summary.json")
    with open(simple_json_path, "w") as f:
        json.dump(simple_json, f, indent=2)

    print(f"✅ 已保存简洁 JSON：{simple_json_path}")

def main(args):
    eval_and_save_problems(args)

if __name__ == "__main__":
    from unit_test_configs import *  
    main(args)
