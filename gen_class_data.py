import json
import os

def decide_label(baseline_pass_total, rag_pass_total, threshold=1):
    return int(rag_pass_total - baseline_pass_total >= threshold)

def build_rag_decision_dataset_from_mixed_ids(
    prompt_base_dir: str,
    baseline_result_dir: str,
    rag_result_dir: str,
    output_path: str,
    start_id: int = 0,
    end_id: int = 5000,
    threshold: int = 1
):
    dataset = []
    missing = 0

    for idx in range(start_id, end_id):
        prompt_id = f"{idx:04d}"
        summary_id = str(idx)

        prompt_path = os.path.join(prompt_base_dir, prompt_id, "question.txt")
        baseline_summary_path = os.path.join(baseline_result_dir, f"{summary_id}_summary.json")
        rag_summary_path = os.path.join(rag_result_dir, f"{summary_id}_summary.json")

        if not os.path.exists(prompt_path):
            print(f"[WARN] Missing prompt: {prompt_path}")
            missing += 1
            continue

        if not os.path.exists(baseline_summary_path):
            print(f"[WARN] Missing baseline summary: {baseline_summary_path}")
            missing += 1
            continue

        if not os.path.exists(rag_summary_path):
            print(f"[WARN] Missing RAG summary: {rag_summary_path}")
            missing += 1
            continue

        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                prompt = f.read().strip()
        except Exception as e:
            print(f"[ERROR] Failed to read prompt {prompt_id}: {e}")
            missing += 1
            continue

        try:
            with open(baseline_summary_path, 'r', encoding='utf-8') as f:
                baseline_summary = json.load(f)
            baseline_pass_total = sum(baseline_summary.get("pass_counts", []))
        except Exception as e:
            print(f"[ERROR] Failed to parse baseline summary {summary_id}: {e}")
            missing += 1
            continue

        try:
            with open(rag_summary_path, 'r', encoding='utf-8') as f:
                rag_summary = json.load(f)
            rag_pass_total = sum(rag_summary.get("pass_counts", []))
        except Exception as e:
            print(f"[ERROR] Failed to parse RAG summary {summary_id}: {e}")
            missing += 1
            continue

        label = decide_label(baseline_pass_total, rag_pass_total, threshold)

        dataset.append({
            "id": prompt_id,
            "prompt": prompt,
            "baseline_passed": baseline_pass_total,
            "rag_passed": rag_pass_total,
            "label": label
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print("\nDataset construction completed")
    print(f"Valid samples: {len(dataset)}")
    print(f"Skipped samples: {missing}")
    print(f"Saved dataset to: {output_path}")

if __name__ == "__main__":
    build_rag_decision_dataset_from_mixed_ids(
        prompt_base_dir="test",
        baseline_result_dir="outputs/1.5B_summaries",
        rag_result_dir="outputs/1.5B_rag_summaries",
        output_path=" ",
        start_id=0,
        end_id=5000,
        threshold=0
    )
