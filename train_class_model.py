import argparse
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import get_linear_schedule_with_warmup
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.optim import AdamW
from tqdm import tqdm
import psutil
import os
import torch.nn as nn
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds//60:.0f}m {seconds%60:.0f}s"
    else:
        return f"{seconds//3600:.0f}h {(seconds%3600)//60:.0f}m"


def print_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"Memory usage: {mem_info.rss / (1024**2):.2f} MB (RSS)")


class RAGDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.texts = [item['prompt'] for item in data]
        self.labels = [item['label'] for item in data]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer.encode_plus(
            self.texts[idx],
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def load_data(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)


def train(args):
    print("Loading training data...")
    train_data = load_data(args.train_path)
    for i, item in enumerate(train_data):
        if not isinstance(item['label'], int) or item['label'] not in [0, 1]:
            print(f"⚠️ Invalid label at index {i}: {item['label']}")

    tokenizer = DebertaV2Tokenizer.from_pretrained(args.model_name)
    model = DebertaV2ForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    model.to(args.device)


    # tokenizer = RobertaTokenizer.from_pretrained(args.model_name)
    # model = RobertaForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    # model.to(args.device)

    train_dataset = RAGDataset(train_data, tokenizer, args.max_length)
    if args.val_path:
        print("Loading validation data from:", args.val_path)
        val_data = load_data(args.val_path)
        val_dataset = RAGDataset(val_data, tokenizer, args.max_length)
    else:
        print("No validation path provided, using 90/10 split from training data.")
        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps
    )

    os.makedirs(args.output_dir, exist_ok=True)
    best_val_acc = 0.0
    best_epoch = -1

    for epoch in range(args.epochs):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            labels = batch['label'].to(args.device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            preds = outputs.logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total

        val_loss, val_correct, val_total = 0, 0, 0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(args.device)
                attention_mask = batch['attention_mask'].to(args.device)
                labels = batch['label'].to(args.device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
                preds = outputs.logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Train Acc = {train_acc:.2%}, Val Loss = {avg_val_loss:.4f}, Val Acc = {val_acc:.2%}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_dir = os.path.join(args.output_dir, "best_model")
            os.makedirs(best_dir, exist_ok=True)
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            print(f"🌟 Best model updated at epoch {epoch+1} with Val Acc = {val_acc:.2%}")

        if (epoch + 1) % args.save_interval == 0:
            save_dir = os.path.join(args.output_dir, f"epoch{epoch+1}")
            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"✅ Saved model at epoch {epoch+1} to {save_dir}")

    if args.epochs % args.save_interval != 0:
        save_dir = os.path.join(args.output_dir, f"e{args.epochs}")
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"✅ Saved final model at epoch {args.epochs} to {save_dir}")

    print(f"🎉 Best Val Acc: {best_val_acc:.2%} at epoch {best_epoch}, saved to {best_dir}")
    print("Training completed.")


def inference(args):

    model_dir = os.path.join(args.output_dir, "best_model")
    tokenizer = DebertaV2Tokenizer.from_pretrained(model_dir)
    model = DebertaV2ForSequenceClassification.from_pretrained(model_dir).to(args.device)
    # tokenizer = RobertaTokenizer.from_pretrained(model_dir)
    # model = RobertaForSequenceClassification.from_pretrained(model_dir).to(args.device)
    model.eval()


    with open(args.test_path, 'r') as f:
        test_prompts = json.load(f)

    results = []
    correct = 0
    total = 0
    correct_ids = []

    for item in tqdm(test_prompts, desc="Running inference"):
        id_val = item.get('id', None)
        if callable(id_val):
            id_val = None

        prompt = item.get('prompt', '')
        true_label = item.get('label', None)


        encoding = tokenizer(prompt, return_tensors='pt', truncation=True, padding='max_length', max_length=args.max_length)
        input_ids = encoding['input_ids'].to(args.device)
        attention_mask = encoding['attention_mask'].to(args.device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            pred_label = torch.argmax(probs, dim=1).item()

        is_correct = pred_label == true_label if true_label is not None else None
        if is_correct:
            correct += 1
            correct_ids.append(id_val)

        if true_label is not None:
            total += 1


        results.append({
            "id": id_val,
            "prompt": prompt,
            "label": pred_label,
            "confidence": probs[0][pred_label].item(),
            "true_label": true_label,
            "correct": is_correct
        })


    out_path = os.path.join(args.output_dir, "result4.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ All inference results saved to {out_path}")


    if total > 0:
        accuracy = correct / total
        print(f"🎯 Accuracy: {correct} / {total} = {accuracy:.4f}")
    else:
        print("⚠️ Warning: No ground-truth labels found for accuracy calculation.")



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'infer'], default='train')
    parser.add_argument('--model_name', type=str, default='')
    parser.add_argument('--train_path', type=str, default='train_class_data.json')
    parser.add_argument('--val_path', type=str, default='test.json')
    parser.add_argument('--test_path', type=str, default='test.json')
    parser.add_argument('--output_dir', type=str, default='rag_classifier_output')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.mode == 'train':
        train(args)
    else:
        inference(args)
