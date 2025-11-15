import json
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

MODEL_PATH = "bge-base-en-v1.5"
JSON_PATH = Path("")
TEST_ROOT = Path("")


class CodeRetriever:
    def __init__(self, batch_size: int = 8):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.model = AutoModel.from_pretrained(MODEL_PATH).to(self.device)
        self.model.eval()
        self.batch_size = batch_size

        self.prompts, self.prompt_embs = self._load_prompts()

    def _load_prompts(self):
        raw_data = json.loads(JSON_PATH.read_text())
        prompts = []
        for key, item in raw_data.items():
            if isinstance(item, dict) and "prompt" in item and "codes" in item:
                raw_prompt = item["prompt"]
                truncated_prompt = raw_prompt.split("-----Input-----")[0].strip()
                prompts.append({
                    "id": key,  
                    "prompt": truncated_prompt,
                    "codes": item["codes"]
                })

        prompt_texts = [p["prompt"] for p in prompts]
        prompt_embs = self._get_embeddings(prompt_texts)
        return prompts, prompt_embs

    def _get_embeddings(self, texts):
        all_embs = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), self.batch_size), desc="Embedding batches"):
                batch_texts = texts[i: i + self.batch_size]
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                outputs = self.model(**inputs)
                hidden_states = outputs.last_hidden_state
                attention_mask = inputs["attention_mask"]

                mask = attention_mask.unsqueeze(-1).float()
                summed = torch.sum(hidden_states * mask, dim=1)
                counts = torch.clamp(mask.sum(dim=1), min=1e-9)
                batch_embs = F.normalize(summed / counts, p=2, dim=1)
                all_embs.append(batch_embs.cpu())

        return torch.cat(all_embs, dim=0)

    def process_directory(self, test_dir: Path, start_idx: int = 0, end_idx: int = None, top_k: int = 5):
        subdirs = sorted(test_dir.iterdir())[start_idx:end_idx]
        print(f"Processing {len(subdirs)} test cases from index {start_idx} to {end_idx or 'end'}...")

        for subdir in tqdm(subdirs, desc="Processing test cases"):
            q_file = subdir / "question.txt"
            if not q_file.exists():
                continue

            question_raw = q_file.read_text()
            question = question_raw.split("-----Input-----")[0].strip()

            question_emb = self._get_embeddings([question])
            sim_scores = F.cosine_similarity(question_emb.cpu(), self.prompt_embs.cpu())


            top_k_scores = torch.topk(sim_scores, k=top_k)
            top_indices = top_k_scores.indices.tolist()
            top_scores = top_k_scores.values.tolist()



            matches = [self.prompts[idx] for idx in top_indices]

            print(f"\n---\nProcessing file: {q_file.name}")
            print(f"Truncated Input Question:\n{question}\n---")
            print("Best Matched Prompts (top-k):")
            for i, m in enumerate(matches):
                print(f"Top {i+1} (score={top_scores[i]:.4f}): {m['prompt'][:80]}...")


            self._append_codes(q_file, matches, top_scores)


    @staticmethod
    def _append_codes(file_path: Path, match_items: list, similarities: list):
        instruction = (
            " "
        )

        with file_path.open("a", encoding="utf-8") as f:
            f.write(instruction)

            for i, (item, score) in enumerate(zip(match_items, similarities)):
                codes = item["codes"]
                if isinstance(codes, str):
                    codes = [codes]

                code = codes[0]
                cleaned_code = "\n".join([line.rstrip() for line in code.split("\n")])

                f.write(f"# === Match {i+1} (Similarity: {score:.4f}, ID: {item['id']}) ===\n")
                f.write(f"{cleaned_code}\n\n")

        print(f"Appended {len(match_items)} code blocks to {file_path.name}")


if __name__ == "__main__":
    START_IDX = 0
    END_IDX = 5000  # or None
    TOP_K = 1  

    retriever = CodeRetriever(batch_size=8)
    retriever.process_directory(TEST_ROOT, start_idx=START_IDX, end_idx=END_IDX, top_k=TOP_K)
