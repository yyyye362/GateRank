# GateRank: Learning When and What to Retrieve for Function-Level Code Generation


## ✨ Introduction

GateRank is an adaptive RAG framework that decides when to retrieve and what to retrieve for code generation. By combining a lightweight retrieval trigger with an execution-guided knowledge selector, it avoids unnecessary retrieval and injects only useful Knowledge—significantly improving executability with lower inference cost.

![./assets/images/ac_overview.png](./assets/images/SmartRAG.jpg)



## 🛠️ Preparations
The code requires some dependencies as specified in requirements.txt. Please follow the relevant libraries to install or run:
```bash
pip install -r requirements.txt
```

## 📚 Datasets

We use two major datasets in our work: APPS and LeetCode Dataset.

APPS
Download link & instructions:
https://github.com/hendrycks/apps
 provides the full dataset (~1.3 GB). 

Also available via Hugging Face:
```
from datasets import load_dataset
ds = load_dataset("codeparrot/apps")
```
We use the *train* split as the training set, and the *test* split as the evaluation set.

LeetCode Dataset
The curated version “LeetCodeDataset” released recently supports robust evaluation and fine-tuning of code models. 
Download via → [data](data/LeetCode) folder.
We similarly use its training portion for fine-tuning, and its test portion for evaluation.



## 🤗 Model
We fine-tune on four models:
  - [Qwen2.5-Coder (1.5B)](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct)
  - [Qwen2.5-Coder (3B)](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct)
  - [DeepSeek-Coder (6.7B)](https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct)
  - [Qwen2.5-Coder (7B)](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct)



## 🧲 Extracting
Two skeletons can be extracted by the following code:
<pre>
python generate_CTS.py
python generate_SAS.py
</pre>


## 🏋️ Finetuning
## 1. Base Model Finetuning
First, We finetune the code generation model on the training splits of APPS and LeetCode to improve its understanding and generation ability for programming tasks:
```
python train_gen_model.py
```
## 2. Retrieval Trigger Training
Second, Using the difference in test-case pass rates between the baseline and RAG outputs, we automatically construct labels to train a trigger that decides when external retrieval is needed:
```
python train_class_model.py  --mode train
```
## 3. Retrieval Ranker Training
Third, When retrieval is triggered, a ranker selects the most useful snippet from multiple candidates to ensure higher-quality retrieval:
```
python train_rank_model.py
```

## 🔍 Retrieval
The retrieval module in GateRank is responsible for extracting external code knowledge candidates used to augment the generation process:
<pre>
python retrieve.py
</pre>

 
## ✨ Generating
Generate results for different fine-tuning model:
<pre>
python train_class_model.py --mode infer
python rank_run.py
python generate.py
</pre>



## 📊 Evaluate
You can run "test_one_solution.sh" to evaluate the functional correctness and pass count of the generated code:
<pre>
cd evaluate/metric
bash test_one_solution.sh
</pre>
