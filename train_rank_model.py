import json
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer, logging
from torch.optim import AdamW
import numpy as np
from tqdm import tqdm
from sklearn.metrics import ndcg_score

os.environ["CUDA_VISIBLE_DEVICES"] = ""

MODEL_NAME = ""
MAX_LENGTH = 510
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
EPOCHS = 10
MARGIN = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.set_verbosity_error()

tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
base_model = RobertaModel.from_pretrained(MODEL_NAME).to(DEVICE)

class CodeRankingModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.codebert = base_model
        self.dropout = nn.Dropout(0.1)
        self.scoring_layer = nn.Linear(base_model.config.hidden_size, 1)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.codebert(input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        cls_embedding = self.dropout(cls_embedding)
        scores = self.scoring_layer(cls_embedding)
        return scores.squeeze(-1)

class CodeRankingDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length):
        self.data = self.load_data(file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def load_data(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        processed_data = []
        for problem in data:
            prompt = problem["prompt"]
            candidates = problem["candidates"]
            positive_indices = [i for i, cand in enumerate(candidates) 
                                if cand.get("label", "").lower() == "positive"]
            negative_indices = [i for i, cand in enumerate(candidates) 
                                if cand.get("label", "").lower() == "negative"]
            if positive_indices and negative_indices:
                processed_data.append({
                    "prompt": prompt,
                    "candidates": [c["code"] for c in candidates],
                    "positive_indices": positive_indices,
                    "negative_indices": negative_indices
                })
        return processed_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        problem = self.data[idx]
        prompt = problem["prompt"]
        candidates = problem["candidates"]
        positive_indices = problem["positive_indices"]
        negative_indices = problem["negative_indices"]
        candidate_encodings = []
        for candidate in candidates:
            encoding = self.tokenizer(
                text=prompt,
                text_pair=candidate,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            candidate_encodings.append(encoding)
        return {
            "problem_idx": idx,
            "candidate_encodings": candidate_encodings,
            "positive_indices": positive_indices,
            "negative_indices": negative_indices
        }

def collate_fn(batch):
    problem_indices = [item["problem_idx"] for item in batch]
    positive_indices_list = [item["positive_indices"] for item in batch]
    negative_indices_list = [item["negative_indices"] for item in batch]
    all_input_ids = []
    all_attention_masks = []
    candidate_counts = []
    for item in batch:
        encodings = item["candidate_encodings"]
        candidate_counts.append(len(encodings))
        for encoding in encodings:
            all_input_ids.append(encoding["input_ids"].squeeze(0))
            all_attention_masks.append(encoding["attention_mask"].squeeze(0))
    input_ids = torch.stack(all_input_ids)
    attention_mask = torch.stack(all_attention_masks)
    return {
        "problem_indices": problem_indices,
        "positive_indices_list": positive_indices_list,
        "negative_indices_list": negative_indices_list,
        "candidate_counts": candidate_counts,
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }

class MultiPositiveContrastiveLoss(nn.Module):
    def __init__(self, margin=MARGIN):
        super().__init__()
        self.margin = margin
    
    def forward(self, scores, candidate_counts, positive_indices_list, negative_indices_list):
        losses = []
        start_idx = 0
        for count, pos_indices, neg_indices in zip(candidate_counts, positive_indices_list, negative_indices_list):
            end_idx = start_idx + count
            problem_scores = scores[start_idx:end_idx]
            if not pos_indices or not neg_indices:
                start_idx = end_idx
                continue
            positive_scores = problem_scores[pos_indices]
            negative_scores = problem_scores[neg_indices]
            for pos_score in positive_scores:
                for neg_score in negative_scores:
                    loss = torch.relu(self.margin - pos_score + neg_score)
                    losses.append(loss)
            start_idx = end_idx
        if losses:
            return torch.mean(torch.stack(losses))
        return torch.tensor(0.0, device=scores.device)

dataset = CodeRankingDataset("triplet_data.json", tokenizer, MAX_LENGTH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

model = CodeRankingModel(base_model).to(DEVICE)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = MultiPositiveContrastiveLoss(margin=MARGIN)

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        candidate_counts = batch["candidate_counts"]
        positive_indices_list = batch["positive_indices_list"]
        negative_indices_list = batch["negative_indices_list"]
        optimizer.zero_grad()
        scores = model(input_ids, attention_mask)
        loss = criterion(scores, candidate_counts, positive_indices_list, negative_indices_list)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    all_ndcg = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            candidate_counts = batch["candidate_counts"]
            positive_indices_list = batch["positive_indices_list"]
            scores = model(input_ids, attention_mask)
            start_idx = 0
            for i, count in enumerate(candidate_counts):
                end_idx = start_idx + count
                problem_scores = scores[start_idx:end_idx].cpu().numpy()
                ideal_ranking = np.zeros(count)
                for pos_idx in positive_indices_list[i]:
                    if pos_idx < count:
                        ideal_ranking[pos_idx] = 1
                if count > 1:
                    k_values = [1, 3, 5, min(10, count)]
                    ndcg_scores = {}
                    for k in k_values:
                        if k <= count:
                            ndcg_scores[k] = ndcg_score([ideal_ranking], [problem_scores], k=k)
                        else:
                            ndcg_scores[k] = 0
                    all_ndcg.append(ndcg_scores)
                start_idx = end_idx
    if all_ndcg:
        avg_ndcg = {}
        k_values = [1, 3, 5, 10]
        for k in k_values:
            valid_scores = [score[k] for score in all_ndcg if k in score]
            avg_ndcg[k] = np.mean(valid_scores) if valid_scores else 0
        return avg_ndcg
    return {k: 0 for k in [1, 3, 5, 10]}

for epoch in range(EPOCHS):
    train_loss = train_epoch(model, dataloader, optimizer, criterion, DEVICE)
    ndcg_scores = evaluate(model, dataloader, DEVICE)
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f}")
    for k, score in ndcg_scores.items():
        print(f"NDCG@{k}: {score:.4f}", end="  ")
    print()

torch.save(model.state_dict(), "triplet_train/pytorch_model.bin")

def rank_candidates(prompt, candidates, model, tokenizer, device, max_length=MAX_LENGTH):
    encodings = []
    for candidate in candidates:
        encoding = tokenizer(
            text=prompt,
            text_pair=candidate,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        encodings.append(encoding)
    input_ids = torch.stack([e["input_ids"].squeeze(0) for e in encodings]).to(device)
    attention_mask = torch.stack([e["attention_mask"].squeeze(0) for e in encodings]).to(device)
    model.eval()
    with torch.no_grad():
        scores = model(input_ids, attention_mask)
    scores = scores.cpu().numpy()
    ranked_candidates = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return ranked_candidates
