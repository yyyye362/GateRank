import json
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
import re

model_path = "bge-base-en-v1.5"
test_file = "LeetCodeDataset-v0.3.1-test.jsonl"
train_file = "LeetCodeDataset-v0.3.1-train.jsonl"
output_file = ""
taskid_file = "/log.txt"
top_k = 3  

device = "cuda" if torch.cuda.is_available() else "cpu"


tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path).to(device)


def get_embedding(texts, batch_size=64):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0]  
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        all_embeddings.append(embeddings.cpu())
    return torch.cat(all_embeddings, dim=0)


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def extract_before_example(text):

    match = re.search(r'(?i)\bExample\s*\d', text)
    if match:
        return text[:match.start()]
    return text

test_data = load_jsonl(test_file)
train_data = load_jsonl(train_file)


train_prompts = [extract_before_example(item["problem_description"]) for item in train_data]
#train_completions = [item["completion"] for item in train_data]
train_completions = [item["response_skeleton"] for item in train_data]


print("Encoding train prompts...")
train_embeddings = get_embedding(train_prompts)


print("Processing test data...")
new_test_data = []
all_retrieved_task_ids = []

for test_item in tqdm(test_data):

    test_prompt = extract_before_example(test_item["problem_description"])


    test_embedding = get_embedding([test_prompt])


    scores = torch.matmul(test_embedding, train_embeddings.T).squeeze(0)  # [N]
    top_indices = torch.topk(scores, top_k).indices.tolist()
    top_scores = torch.topk(scores, top_k).values.tolist()


    retrieved = [
        {"task_id": train_data[i]["task_id"], "score": float(top_scores[idx])}
        for idx, i in enumerate(top_indices)
    ]

    all_retrieved_task_ids.append({
        "test_task_id": test_item["task_id"],
        "retrieved": retrieved
    })

    fewshot_block = (
        “ ”
    )

    for idx, i in enumerate(top_indices, start=1):
        fewshot_block += f"\n### Example {idx}:\n```python\n{train_completions[i]}\n```\n"

    # if "### Format:" in test_item["query"]:
    #     test_item["query"] = test_item["query"].replace("### Format:", fewshot_block + "\n### Format:")
    # else:
    #     test_item["query"] += "\n\n" + fewshot_block



    test_item["query"] += fewshot_block
    new_test_data.append(test_item)



with open(output_file, "w", encoding="utf-8") as f:
    for item in new_test_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


with open(taskid_file, "w", encoding="utf-8") as f:
    for record in all_retrieved_task_ids:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"新测试文件已保存到: {output_file}")
print(f"检索到的 task_id + score 已保存到: {taskid_file}")