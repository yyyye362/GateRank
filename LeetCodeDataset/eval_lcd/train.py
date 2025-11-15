import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
from peft import get_peft_model, LoraConfig, TaskType
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,1"


model_path = "/home/yzj/model/deepseek-coder-6.7b-instruct"
data_path = "/home/yzj/LeetCodeDataset-main/data/v0.3.1-train.jsonl"
output_dir = "/home/yzj/LeetCodeDataset-main/model/deepseek6.7B-sft1"


tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token


n_gpus = torch.cuda.device_count()
print(f"检测到 {n_gpus} 张 GPU, 启动模型并行...")

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map="auto",       
)


lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  
    inference_mode=False,          
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    #target_modules=["q_proj", "v_proj", "k_proj", "o_proj"] 
)


model = get_peft_model(model, lora_config)


def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]

raw_data = load_jsonl(data_path)

dataset = Dataset.from_list([
    {"prompt": item["query"], "response": item["response"]}
    for item in raw_data if item["query"].strip() and item["response"].strip()
])


def preprocess(example):
    full_text = example["prompt"] + example["response"]
    tokenized = tokenizer(full_text, truncation=True, max_length=1024, padding="max_length")
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(preprocess, remove_columns=["prompt", "response"])


training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=5,
    fp16=True,
    logging_steps=20,
    save_strategy="steps",
    save_steps=2641,
    save_total_limit=1,
    learning_rate=1e-5,
    warmup_steps=100,
    weight_decay=0.01,
    report_to="none",
    ddp_find_unused_parameters=False,   
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=default_data_collator,
)


trainer.train()


model.save_pretrained(output_dir)
print(f"LoRA 适配器已保存到 {output_dir}")
