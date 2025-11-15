import json
import os
import logging
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from tqdm import tqdm

from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

class MultiTaskDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_source_length=1600, max_target_length=512):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        logger.info(f"Loaded dataset with {len(self.data)} samples from {data_path}")
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_src_len = max_source_length
        self.max_tgt_len = max_target_length
        self.max_seq_len = self.max_src_len + self.max_tgt_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = '\nQUESTION:\n' + item['prompt'].strip()
        if item['label'] == 1 and item.get('rag_context'):
            prompt += '\n[CONTEXT]\n' + item['rag_context'].strip()
        cls_label = torch.tensor(item['label'], dtype=torch.long)
        solutions = item.get('solutions', [])
        sampled_code = random.choice(solutions).strip() if solutions else ""
        full_text = prompt + '\nANSWER:\n' + sampled_code + self.tokenizer.eos_token
        prompt_enc = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_src_len,
            padding=False,
            return_tensors='pt'
        )
        prompt_input_ids = prompt_enc.input_ids.squeeze(0)
        full_enc = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_seq_len,
            padding='max_length',
            return_tensors='pt'
        )
        input_ids = full_enc.input_ids.squeeze(0)
        attention_mask = full_enc.attention_mask.squeeze(0)
        labels = input_ids.clone()
        labels[:len(prompt_input_ids) + 1] = -100
        return {
            'prompt_input_ids': prompt_input_ids,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'cls_label': cls_label
        }

class MultiTaskModel(nn.Module):
    def __init__(
        self,
        model_name='Qwen2.5-Coder-1.5B-Instruct',
        num_labels=2,
        lora_r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        use_gradient_checkpointing=True
    ):
        super().__init__()
        try:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
                device_map="auto",
                max_memory={i: "24000MiB" for i in range(torch.cuda.device_count())},
                use_cache=False
            )
            logger.info(f"Loaded base model: {model_name} with device map: {self.base_model.hf_device_map}")
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            raise
        self.base_model = prepare_model_for_kbit_training(self.base_model)
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            modules_to_save=["lm_head", "embed_tokens"]
        )
        self.model = get_peft_model(self.base_model, peft_config)
        hidden_size = self.model.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_labels)
        ).to(f"cuda:0")
        if use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing")

    def forward(self, prompt_input_ids, input_ids, attention_mask, labels, cls_label):
        start_device = list(self.model.hf_device_map.values())[0]
        prompt_input_ids = prompt_input_ids.to(start_device)
        input_ids = input_ids.to(start_device)
        attention_mask = attention_mask.to(start_device)
        labels = labels.to(start_device)
        cls_label = cls_label.to(start_device)
        cls_output = self.model(
            input_ids=prompt_input_ids,
            output_hidden_states=True,
            return_dict=True
        )
        last_hidden = cls_output.hidden_states[-1]
        last_hidden = last_hidden.to(f"cuda:0")
        pooled_output = last_hidden.mean(dim=1)
        cls_logits = self.classifier(pooled_output)
        gen_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        gen_loss = gen_output.loss
        cls_loss = F.cross_entropy(cls_logits, cls_label.to(f"cuda:0"))
        return cls_logits, gen_loss, cls_loss

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        self.model.save_pretrained(save_directory)
        torch.save(self.classifier.state_dict(), os.path.join(save_directory, "classifier.bin"))
        logger.info(f"Model saved to {save_directory}")

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    def pad_sequence(sequences, max_len=None, padding_value=0):
        if max_len is None:
            max_len = max(len(s) for s in sequences)
        padded = torch.full((len(sequences), max_len), padding_value, dtype=sequences[0].dtype)
        for i, seq in enumerate(sequences):
            padded[i, :len(seq)] = seq
        return padded
    prompt_input_ids = [b['prompt_input_ids'] for b in batch]
    input_ids = [b['input_ids'] for b in batch]
    attention_mask = [b['attention_mask'] for b in batch]
    labels = [b['labels'] for b in batch]
    cls_labels = [b['cls_label'] for b in batch]
    max_prompt_len = max(len(ids) for ids in prompt_input_ids)
    max_seq_len = max(len(ids) for ids in input_ids)
    return {
        'prompt_input_ids': pad_sequence(prompt_input_ids, max_prompt_len, 0),
        'input_ids': pad_sequence(input_ids, max_seq_len, 0),
        'attention_mask': pad_sequence(attention_mask, max_seq_len, 0),
        'labels': pad_sequence(labels, max_seq_len, -100),
        'cls_label': torch.stack(cls_labels)
    }

def train(
    train_path,
    model_name,
    batch_size,
    epochs,
    lr,
    cls_weight,
    gen_weight,
    output_dir,
    lora_r,
    lora_alpha,
    lora_dropout,
    grad_accum_steps,
    max_grad_norm,
    warmup_ratio,
    log_interval,
    max_source_length,
    max_target_length
):
    device_count = torch.cuda.device_count()
    logger.info(f"Found {device_count} GPU devices")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        logger.error(f"Error loading tokenizer: {str(e)}")
        raise
    try:
        train_dataset = MultiTaskDataset(train_path, tokenizer, max_source_length, max_target_length)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4
        )
        logger.info(f"Created training dataloader with {len(train_loader)} batches")
    except Exception as e:
        logger.error(f"Error creating training dataset: {str(e)}")
        raise
    try:
        model = MultiTaskModel(
            model_name=model_name,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            use_gradient_checkpointing=False
        )
        logger.info(f"Initialized model with LoRA (r={lora_r}, alpha={lora_alpha})")
        logger.info(f"Model device map: {model.model.hf_device_map}")
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")
        raise
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs // grad_accum_steps
    warmup_steps = int(warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    logger.info(f"Total training steps: {total_steps} (warmup: {warmup_steps})")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_cls_loss, total_gen_loss = 0, 0, 0
        total_samples, correct_cls = 0, 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
        for step, batch in progress_bar:
            if batch is None:
                logger.warning(f"Skipping empty batch at step {step}")
                continue
            prompt_input_ids = batch['prompt_input_ids']
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            cls_labels = batch['cls_label']
            cls_logits, gen_loss, cls_loss = model(
                prompt_input_ids=prompt_input_ids,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                cls_label=cls_labels
            )
            loss = cls_weight * cls_loss + gen_weight * gen_loss
            loss.backward()
            batch_size = input_ids.size(0)
            total_loss += loss.item() * batch_size
            total_cls_loss += cls_loss.item() * batch_size
            total_gen_loss += gen_loss.item() * batch_size
            preds = torch.argmax(cls_logits, dim=-1)
            correct_cls += (preds.cpu() == cls_labels.cpu()).sum().item()
            total_samples += batch_size
            if (step + 1) % grad_accum_steps == 0 or (step + 1 == len(train_loader)):
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            if (step + 1) % log_interval == 0:
                progress_bar.set_postfix({
                    'loss': loss.item(),
                    'cls_loss': cls_loss.item(),
                    'gen_loss': gen_loss.item()
                })
        avg_loss = total_loss / total_samples
        avg_cls_loss = total_cls_loss / total_samples
        avg_gen_loss = total_gen_loss / total_samples
        cls_acc = correct_cls / total_samples
        logger.info(f"Epoch {epoch}/{epochs} | Train Loss: {avg_loss:.4f} | "
                    f"CLS Loss: {avg_cls_loss:.4f} | GEN Loss: {avg_gen_loss:.4f} | "
                    f"CLS Acc: {cls_acc:.4f}")
        epoch_save_path = os.path.join(output_dir, f'epoch_{epoch}')
        model.save_pretrained(epoch_save_path)
        logger.info(f"Saved epoch checkpoint to {epoch_save_path}")
    logger.info("Training completed!")

if __name__ == '__main__':
    output_dir = ' '
    os.makedirs(output_dir, exist_ok=True)
    train(
        train_path=' ',
        model_name=' ',
        batch_size=1,
        epochs=10,
        lr=2e-5,
        cls_weight=0.7,
        gen_weight=0.3,
        output_dir=output_dir,
        lora_r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        grad_accum_steps=4,
        max_grad_norm=1.0,
        warmup_ratio=0.1,
        log_interval=10,
        max_source_length=1600,
        max_target_length=512
    )
