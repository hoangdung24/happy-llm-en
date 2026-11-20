---
title: "Chapter 6: Training LLMs in Practice"
description: "Training LLMs in practice: distributed training, mixed precision, data pipelines, checkpointing, and monitoring a real run."
---

Chapter 5 built the LLaMA2 model from scratch and implemented its full pretraining and fine-tuning pipeline. This chapter covers the practice of the LLM training pipeline. It shows how to use mainstream LLM frameworks to train models and optimize performance.

> Reading advice: read [Chapter 6 Practice Guide](./readme.md) and [Learning and Environment Setup](../学习与环境准备.md) before you start the practice in this chapter. The main text of this chapter covers three threads: Pretrain, SFT, and PEFT. Preference alignment is an advanced topic. Section 6.4 outlines the learning route and provides supplementary materials.

## 6.1 Model Pretraining

The previous chapter covered the model architecture and the training process of LLMs. We hand-implemented the LLaMA architecture and the full Pretrain and SFT pipelines from scratch. That work gives a deeper view of LLM principles and training details. In practice, hand-written LLM training has these problems:

- Hand-implementing the LLM architecture takes a lot of work. You cannot keep up with new architectural innovations in real time.
- An LLM implemented from scratch does not support multi-GPU distributed training well. Training efficiency is low.
- It is incompatible with existing pretrained LLMs, so it cannot use pretrained model parameters.

This chapter introduces Transformers, the mainstream training framework in the LLM field. With frameworks such as DeepSpeed for distribution and peft for efficient fine-tuning, we practice the full Pretrain and SFT pipeline with transformers. This approach matches the industry's mainstream LLM technical solutions.

### 6.1.1 Framework Introduction

Transformers is an NLP framework developed by Hugging Face. Its modular design gives unified support for hundreds of mainstream model architectures, including BERT, GPT, LLaMA, T5, and ViT. With Transformers, developers no longer need to repeatedly implement basic network structures. You can load any pretrained model with a single call via the AutoModel class. Figure 6.1 shows the homepage of the Hugging Face Transformers course:

<div align='center'>
    <img src="/images/6-images/1-1.webp" alt="Figure 6.1 Hugging Face Transformers" width="1544" height="678" loading="lazy" decoding="async" />
    <p>Figure 6.1 Hugging Face Transformers</p>
</div>

The framework also includes the Trainer class. Trainer encapsulates the core logic of distributed training. It supports several distributed training strategies, such as PyTorch native DDP, DeepSpeed, and Megatron-LM. You configure the training parameters only. Then you get hybrid parallel training that combines data parallelism, model parallelism, and pipeline parallelism. This setup supports efficient training of models with tens of billions of parameters on an 8-GPU A100 cluster. Components such as SavingPolicy and LoggingCallback manage the training process automatically. Trainer also integrates with frameworks such as DeepSpeed, peft, wandb, and Swanlab. You connect them through parameter settings, so LLM training is fast.

The Hugging Face community matters even more for NLP researchers in the LLM era. It sits on top of the Transformers framework. It opens hundreds of millions of pretrained model parameters and 250,000+ datasets of various types. Frameworks such as Transformers, Datasets, and Evaluate join pretrained models, datasets, and evaluation functions. Developers can use any pretrained model with one call. They can also develop and deploy their own models on the basis of open-source models and datasets.

<div align='center'>
    <img src="/images/6-images/1-2.webp" alt="Figure 6.2 The Hugging Face Transformers model community" width="1600" height="831" loading="lazy" decoding="async" />
    <p>Figure 6.2 The Hugging Face Transformers model community</p>
</div>

In the LLM era, model architecture changes and fresh pretraining are less common. Business applications now use pretrained LLMs for Post-Training and SFT. Pretrained models are also large, so integration with distributed training frameworks such as DeepSpeed is now an essential skill for NLP model training. As a result, Transformers has become the mainstream framework for NLP in both academia and industry. Enterprise development and scientific research both pick Transformers first for model implementation. New open-source LLMs such as DeepSeek and Qwen also publish their pretrained weights and invocation demos in the Transformers community at release. With the Transformers framework, LLM training and development are fast and convenient, and they deliver industrial-grade output. Next, we use the Transformers framework to implement LLM Pretrain and SFT.

### 6.1.2 Initializing the LLM

You can use the AutoModel class from transformers to initialize an already-implemented model directly. For any pretrained model, its parameters contain the model's configuration information. To train an LLM from scratch, initialize it directly with an existing model architecture. Here, we use the model architecture of [Qwen-2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B/tree/main) as an example:

<div align='center'>
    <img src="/images/6-images/1-3.webp" alt="Figure 6.3 Qwen-2.5-1.5B" width="1600" height="818" loading="lazy" decoding="async" />
    <p>Figure 6.3 Qwen-2.5-1.5B</p>
</div>

This page shows the Qwen-2.5-1.5B model parameters in the Hugging Face community. The `config.json` file holds the model's configuration information: the model architecture, hidden size, number of layers, and so on. Figure 6.4 shows the file:

<div align='center'>
    <img src="/images/6-images/1-4.webp" alt="Figure 6.4 The Qwen-2.5-1.5B config.json file" width="1600" height="818" loading="lazy" decoding="async" />
    <p>Figure 6.4 The Qwen-2.5-1.5B config.json file</p>
</div>

You can reuse this model's configuration information to initialize a Qwen-2.5-1.5B model for training. You can also modify the configuration information, for example the hidden size or the number of attention heads, to customize a model architecture. Hugging Face provides Python tools to download the model parameters you want:

```python
import os
# Set the environment variable; here we use the HuggingFace mirror site
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# Download the model
os.system('huggingface-cli download --resume-download Qwen/Qwen2.5-1.5B --local-dir your_local_dir')
```

As Figure 6.5 shows, "Qwen/Qwen2.5-1.5B" here is the identifier of the model to download. For other models, copy the model name directly from Hugging Face:

<div align='center'>
    <img src="/images/6-images/1-5.webp" alt="Figure 6.5 Model download identifier" width="1600" height="841" loading="lazy" decoding="async" />
    <p>Figure 6.5 Model download identifier</p>
</div>

After download, use the AutoConfig class to load the downloaded configuration file directly:

```python
# Load the predefined model parameters - here using Qwen-2.5-1.5B as an example
# Load using the Config class from transformers
from transformers import AutoConfig

# Local path of the downloaded parameters
model_path = "qwen-1.5b"
config = AutoConfig.from_pretrained(model_name_or_path)
```

You can also customize the configuration file and then load it in the same way. Use the AutoModel class to generate the corresponding model from the loaded configuration object:

```python
# Generate a defined model from this configuration
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_config(config,trust_remote_code=True)
```

LLMs generally use the CausalLM architecture, so we use the AutoModelForCausalLM class for loading here. For classification-task training, use the AutoModelForSequenceClassification class instead. Figure 6.6 shows that this model's architecture matches the defined configuration file:

<div align='center'>
    <img src="/images/6-images/1-6.webp" alt="Figure 6.6 Model architecture output" width="913" height="698" loading="lazy" decoding="async" />
    <p>Figure 6.6 Model architecture output</p>
</div>

This model is a Qwen-2.5-1.5B model initialized from scratch. In general, we rarely pretrain an LLM initialized from scratch. The more common practice is to load a pretrained LLM's weights and continue training on your own corpus. Here, we also show how to initialize a pretrained model from downloaded model parameters.

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(model_name_or_path,trust_remote_code=True)
```

Load it directly with the from_pretrained method. Here, model_name_or_path is the local path of the downloaded parameters.

We also need a tokenizer. Here, use the tokenizer parameters of Qwen-2.5-1.5B directly:

```python
# Load a pretrained tokenizer
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
```

The loaded tokenizer tokenizes any text directly.

### 6.1.3 Pretraining Data Processing

As in Chapter 5, we use the Mobvoi Seq Monkey open-source dataset for pretraining. Download and extract it as in Chapter 5. Hugging Face's datasets library is a third-party library that pairs with the transformers framework for data download and processing. Use the load_dataset function of datasets to load the pretraining data:

```python
# Load the pretraining data
from datasets import load_dataset

ds = load_dataset('json', data_files='/mobvoi_seq_monkey_general_open_corpus.jsonl')
```

The dataset is large, so loading can take a long time or run out of memory. Split off a portion of the pretraining dataset for early testing. The loaded ds is a DatasetDict object. The loaded data is saved by default under the `train` key. Inspect it with the following code:

```python
ds["train"][0]
```

<div align='center'>
    <img src="/images/6-images/1-7.webp" alt="Figure 6.7 Dataset preview" width="1600" height="374" loading="lazy" decoding="async" />
    <p>Figure 6.7 Dataset preview</p>
</div>

View the dataset's features, that is, its columns, through the features attribute. Here we need to save the dataset's column names. In later processing, we tokenize the text and then remove the original text:

```python
# View the features
column_names = list(ds["train"].features)
# columnes_name:["text"]
```

Next, use the loaded tokenizer to process the dataset. Use the map function for batch processing:

```python
# Tokenize the dataset
def tokenize_function(examples):
    # Tokenize using the preloaded tokenizer
    output = tokenizer([item for item in examples["text"]])
    return output

# Batch processing
tokenized_datasets = ds.map(
    tokenize_function,
    batched=True,
    num_proc=10,
    remove_columns=column_names,
    load_from_cache_file=True,
    desc="Running tokenizer on dataset",
)
```

The processed dataset has two columns, 'input_ids' and 'attention_mask'. They hold the tokenized numeric sequences of the text and the attention mask. The mask shows whether padding was applied. The map method removes the original 'text' column through the remove_columns parameter. Training no longer uses that column.

Pretraining is generally a CLM task. Learning the sequence semantics of multiple samples at once does not affect model performance. The training data is also large and training takes long, so the requirements on training efficiency are high. During pretraining, we usually concatenate multiple text segments and process them into uniform text blocks. Each text block is then used for training. Here, we implement a concatenation function that joins text into blocks of 2048 tokens. We use the map method for batch processing:

```python
# Pretraining generally concatenates text into fixed-length text segments
from itertools import chain

# Here we take the block length to be 2048
block_size = 2048

def group_texts(examples):
    # Concatenate the text segments
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    # Compute the total length after concatenation
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # If the length is too long, chunk it
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    # For the CLM task, labels and inputs are identical
    result["labels"] = result["input_ids"].copy()
    return result

# Batch processing
lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    num_proc=10,
    load_from_cache_file=True,
    desc=f"Grouping texts in chunks of {block_size}",
    batch_size = 40000,
)
train_dataset = lm_datasets["train"]
```

The resulting train_dataset is a pretraining dataset ready for CLM Pretrain. Each sample is 2048 tokens long.

### 6.1.4 Training with the Trainer

Next, we train with the Trainer class provided by transformers. Trainer encapsulates the model training logic and includes efficiency optimizations and visualization. It makes LLM training fast and convenient.

First, configure the training hyperparameters. Use the TrainingArguments class to instantiate a parameter object:

```python
from transformers import TrainingArguments
# Configure training arguments

training_args = TrainingArguments(
    output_dir="output",# Output path for training artifacts
    per_device_train_batch_size=4,# Training batch_size
    gradient_accumulation_steps=4,# Gradient accumulation steps; effective bs = set bs * accumulation steps
    logging_steps=10,# Step interval for printing loss
    num_train_epochs=1,# Number of training epochs
    save_steps=100, # Step interval for saving model parameters
    learning_rate=1e-4,# Learning rate
    gradient_checkpointing=True# Enable gradient checkpointing
)
```

Then instantiate a trainer object from the initialized model, tokenizer, and training_args. Pass in the processed training dataset:

```python
from transformers import Trainer, default_data_collator
from torchdata.datapipes.iter import IterableWrapper

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset= IterableWrapper(train_dataset),
    eval_dataset= None,
    tokenizer=tokenizer,
    # Defaults to the MLM collator; use the CLM collator
    data_collator=default_data_collator
)
```

Then call the train method. It trains and saves according to the configured training hyperparameters:

```python
trainer.train()
```

> Note: the above code is stored in the `./code/pretrain.ipynb` file.

### 6.1.5 Distributed Training with DeepSpeed

Pretraining is large in scale and long in duration. Do not run it in a Jupyter Notebook, because it is prone to interruption. The large scale also usually requires multi-GPU distributed training. Otherwise training takes too long. Based on the above code, we show how to use the DeepSpeed framework for distributed training and complete industry-usable LLM Pretrain.

Long-running training generally uses a bash script to set the hyperparameters. The script then launches a pre-written Python script to run the training. We use a Python script (`./code/pretrain.py`) to implement the full training pipeline.

First import the required third-party libraries:

```python
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from torchdata.datapipes.iter import IterableWrapper
from itertools import chain
import deepspeed
from typing import Optional,List

import datasets
import pandas as pd
import torch
from datasets import load_dataset
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
import datetime
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
import swanlab
```

First, define several hyperparameter classes to handle the hyperparameter values set in the sh script. Transformers already has the TrainingArguments class, which includes the essential training hyperparameters. We only need to define the hyperparameters not included in TrainingArguments. These are mainly the model hyperparameters, defined in ModelArguments, and the data hyperparameters, defined in DataTrainingArguments:

```python
# Hyperparameter classes
@dataclass
class ModelArguments:
    """
    Arguments about the model
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Used for post-training: path to the pretrained model parameters"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Used for pretraining: path to the Config file"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Used for pretraining: path to the Tokenizer"}
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Data type used for model training; bfloat16 recommended"
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments about training
    """

    train_files: Optional[List[str]]  = field(default=None, metadata={"help": "Path to the training data"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Configured text block length"
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "Number of threads used for preprocessing."},
    )
```

Then define a main function to wrap the training process described above. First, use the HfArgumentParser tool provided by transformers to load the hyperparameters set in the sh script:

```python
# Load script arguments
parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()
```

Large-scale training uses logs to record the training process. Direct print is not a good choice, because key training information can be lost. Here, use the built-in logging library of Python. First, configure logging:

```python
# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Set the log level to INFO
transformers.utils.logging.set_verbosity_info()
log_level = training_args.get_process_log_level()
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()
```

Here the log level is INFO. The logging library has five levels: DEBUG, INFO, WARNING, ERROR, and CRITICAL. The set level outputs messages at that level and above only. Once you configure logging, use logger directly wherever logging is needed. Specify the log level when recording, for example:

```python
# Record overall training status
logger.warning(
    f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
    + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
)
logger.info(f"Training/evaluation parameters {training_args}")
```

The script below does not cover logging in more detail.

Large-scale training often runs into interruptions. Training generally saves checkpoints at fixed intervals. After an interruption, resume training from the most recent checkpoint. First check whether an old checkpoint exists, and resume training from it:

```python
# Check for checkpoints
last_checkpoint = None
if os.path.isdir(training_args.output_dir):
    # Automatically detect using transformers' built-in get_last_checkpoint
    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
        raise ValueError(
            f"Output path ({training_args.output_dir}) is not empty "
        )
    elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(
            f"Resuming training from {last_checkpoint}"
        )
```

Next, initialize the model in the way introduced above. Here, the code wraps from-scratch initialization and initialization from an existing pretrained model together:

```python
# Initialize the model
if model_args.config_name is not None:
    # from scrach
    config = AutoConfig.from_pretrained(model_args.config_name)
    logger.warning("You are initializing a model from scratch")
    logger.info(f"Model parameter configuration path: {model_args.config_name}")
    logger.info(f"Model parameters: {config}")
    model = AutoModelForCausalLM.from_config(config,trust_remote_code=True)
    n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    logger.info(f"Pretraining a new model - Total size={n_params/2**20:.2f}M params")
elif model_args.model_name_or_path is not None:
    logger.warning("You are initializing a pretrained model")
    logger.info(f"Model parameter path: {model_args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path,trust_remote_code=True)
    n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    logger.info(f"Inheriting a pretrained model - Total size={n_params/2**20:.2f}M params")
else:
    logger.error("config_name and model_name_or_path cannot both be empty")
    raise ValueError("config_name and model_name_or_path cannot both be empty")
```

Then load the tokenizer and process the pretraining data in the same way. This part is identical to the code above, so we do not repeat it here. Check the details in the code. Then use the Trainer for training:

```python
logger.info("Initializing Trainer")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset= IterableWrapper(train_dataset),
    tokenizer=tokenizer,
    data_collator=default_data_collator
)

# Load from checkpoint
checkpoint = None
if training_args.resume_from_checkpoint is not None:
    checkpoint = training_args.resume_from_checkpoint
elif last_checkpoint is not None:
        checkpoint = last_checkpoint

logger.info("Starting training")
train_result = trainer.train(resume_from_checkpoint=checkpoint)
trainer.save_model() 
```

We checked for existing checkpoints above. This code uses resume_from_checkpoint to resume training from a checkpoint.

Large-scale training must monitor training progress and the downward trend of the loss. The script uses swanlab as the training monitoring tool. Swanlab is initialized at the beginning of the script:

```python
# Initialize SwanLab
swanlab.init(project="pretrain", experiment_name="from_scrach")
```

After training starts, the terminal prints the swanlab monitoring URL. Click it to observe training progress. This chapter does not cover swanlab details. Consult the relevant documentation.

The above code is complete. Now use a sh script (`./code/pretrain.sh`) to set the hyperparameter values. The script launches training through DeepSpeed. This step gives efficient multi-GPU distributed training:

```bash
# Set visible GPUs
CUDA_VISIBLE_DEVICES=0,1

deepspeed pretrain.py \
    --config_name autodl-tmp/qwen-1.5b \
    --tokenizer_name autodl-tmp/qwen-1.5b \
    --train_files autodl-tmp/dataset/pretrain_data/mobvoi_seq_monkey_general_open_corpus_small.jsonl \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --do_train \
    --output_dir autodl-tmp/output/pretrain \
    --evaluation_strategy  no \
    --learning_rate 1e-4 \
    --num_train_epochs 1 \
    --warmup_steps 200 \
    --logging_dir autodl-tmp/output/pretrain/logs \
    --logging_strategy steps \
    --logging_steps 5 \
    --save_strategy steps \
    --save_steps 100 \
    --preprocessing_num_workers 10 \
    --save_total_limit 1 \
    --seed 12 \
    --block_size 2048 \
    --bf16 \
    --gradient_checkpointing \
    --deepspeed ./ds_config_zero2.json \
    --report_to swanlab
    # --resume_from_checkpoint ${output_model}/checkpoint-20400 \
```

After you install the DeepSpeed third-party library, launch multi-GPU training directly with the deepspeed command. The script commands above mainly set the values of the various hyperparameters. Use them as a reference. Chapter 4 covered the principles of DeepSpeed distributed training and the ZeRO stage settings. Here, we train with ZeRO-2. The file `ds_config_zero.json` holds the DeepSpeed configuration:

```json
{
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "bf16": {
        "enabled": "auto"
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto"
        }
    },

    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto"
        }
    },

    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "none",
            "pin_memory": true
        },
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": true
    },

    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 100,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
```

Finally, run the `pretrain.sh` script in the terminal with bash to start training.

## 6.2 Supervised Fine-Tuning of the Model

The previous section showed how to use the Transformers framework for fast model pretraining. This part builds on that content. It shows how to use the Transformers framework for supervised fine-tuning on a pretrained model.

### 6.2.1 Pretrain VS SFT

First, review the core difference between pretraining an LLM and supervised fine-tuning it. As Chapter 4 said, mature LLMs today are generally trained through the three stages Pretrain-SFT-RLHF. In the Pretrain stage, the model models massive unlabeled text in a self-supervised manner. It learns text semantic rules and the world knowledge contained in the text. In the SFT stage, the pretrained model is generally instruction-tuned. The model is trained to complete tasks according to user instructions. Then it can follow user instructions and plan, act, and produce output based on them. Both Pretrain and SFT use CLM modeling. The core difference is the data. Pretrain trains on massive unlabeled text, where the model performs the "predict the next token" task on the text. SFT uses constructed paired instruction data, where the model models the output that follows the input instruction. Pretrain computes loss over the entire text, so the model must model and predict the whole text. SFT computes loss only on the output and does not compute loss on the instruction part.

Compared with the Pretrain code from the previous section, the SFT part only modifies the data processing step. You implement the construction that converts paired instruction data into training samples. The rest follows the same implementation logic as Pretrain. The code script for this part is `./code/finetune.py`.

### 6.2.2 Fine-tuning Data Processing

As in Chapter 5, we use the BelleGroup dataset open-sourced by Beike for SFT here.

During SFT, we define a Chat Template. It specifies how to convert conversation data into a text sequence that the model can model and fit. When we use a model that has undergone SFT for downstream task fine-tuning, we generally check that model's Chat Template and adapt to it. This step protects the instruction-following ability it learned during SFT. We use a Pretrain model for SFT here, so we define a custom Chat Template. We used the Qwen-2.5-1.5B model architecture for Pretrain, so we continue to use the Qwen-2.5 Chat Template here. If you do not have enough resources to pretrain the model, you can also use the official Qwen-2.5-1.5B model as the base model for SFT.

First, define several special tokens. Special tokens play special roles during model fitting: the beginning of the text sequence (BOS), the end of the text sequence (EOS), and newline characters. Defining special tokens helps avoid semantic confusion during model fitting:

```python

# Different tokenizers need special definitions
# BOS
im_start = tokenizer("<|im_start|>").input_ids
# EOS
im_end = tokenizer("<|im_end|>").input_ids
# PAD
IGNORE_TOKEN_ID = tokenizer.pad_token_id
# Newline character
nl_tokens = tokenizer('\n').input_ids
# Role identifiers
_system = tokenizer('system').input_ids + nl_tokens
_user = tokenizer('human').input_ids + nl_tokens
_assistant = tokenizer('assistant').input_ids + nl_tokens
```

The Chat Template of the Qwen series generally has three conversation roles: System, User, and Assistant. System is the system prompt. It activates the model's capabilities. Its default is "You are a helpful assistant.", and it generally does not change during SFT. User is the prompt given by the user. Here, the conversation role in the dataset is "human", so we changed "user" to "human". Assistant is the reply given by the LLM. It is the text the model must fit during SFT.

This dataset is a multi-turn conversation dataset. Next, concatenate the multi-turn conversations. Join the turns into a single text sequence:

```python
# Concatenate multi-turn conversations
input_ids, targets = [], []
# Multiple samples
for i in tqdm(range(len(sources))):
    # source is one multi-turn conversation sample
    source = sources[i]
    # Start from the user
    if source[0]["from"] != "human":
        source = source[1:]
    # The input and the output respectively
    input_id, target = [], []
    # system: 【BOS】system\nYou are a helpful assistant.【EOS】\n
    system = im_start + _system + tokenizer(system_message).input_ids + im_end + nl_tokens
    input_id += system
    # The system part is not fitted
    target += im_start + [IGNORE_TOKEN_ID] * (len(system)-3) + im_end + nl_tokens
    assert len(input_id) == len(target)
    # Concatenate in order
    for j, sentence in enumerate(source):
        # sentence is one turn of the conversation
        role = roles[sentence["from"]]
        # user：<|im_start|>human\ninstruction【EOS】\n
        # assistant：<|im_start|>assistant\nresponse【EOS】\n
        _input_id = tokenizer(role).input_ids + nl_tokens + \
            tokenizer(sentence["value"]).input_ids + im_end + nl_tokens
        input_id += _input_id
        if role == '<|im_start|>human':
            # The user part is not fitted
            _target = im_start + [IGNORE_TOKEN_ID] * (len(_input_id)-3) + im_end + nl_tokens
        elif role == '<|im_start|>assistant':
            # The assistant part is fitted
            _target = im_start + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids) + \
                _input_id[len(tokenizer(role).input_ids)+1:-2] + im_end + nl_tokens
        else:
            print(role)
            raise NotImplementedError
        target += _target
    assert len(input_id) == len(target)
    # Finally, pad
    input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
    target += [IGNORE_TOKEN_ID] * (max_len - len(target))
    input_ids.append(input_id[:max_len])
    targets.append(target[:max_len])
```

The code above follows the Qwen Chat Template logic. You can modify it to your own preferences. Its key point is that the User's text does not need to be fitted. So the code masks the User's corresponding text content in targets with IGNORE_TOKEN_ID. The Assistant's corresponding text content is the original text, and the code computes loss for it. For mainstream LLMs today, IGNORE_TOKEN_ID is generally set to -100.

After concatenation, convert the tokenized numeric sequences into `Torch.tensor`. Then assemble them into the dictionary that the Dataset returns:

```python
input_ids = torch.tensor(input_ids)
targets = torch.tensor(targets)

return dict(
    input_ids=input_ids,
    labels=targets,
    attention_mask=input_ids.ne(tokenizer.pad_token_id),
)
```

The above processing logic is complete. Next, define a custom Dataset class that calls this logic to process the data:

```python
class SupervisedDataset(Dataset):

    def __init__(self, raw_data, tokenizer, max_len: int):
        super(SupervisedDataset, self).__init__()
        # Load and preprocess the data
        sources = [example["conversations"] for example in raw_data]
        # preprocess is the data preprocessing logic defined above
        data_dict = preprocess(sources, tokenizer, max_len)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )
```

This class inherits from Torch's Dataset class and works directly in the Trainer. The data processing is done. Now use the script from the previous section and modify the data processing logic only. The model training that follows stays the same. The main function logic is attached here:

```python
# Load script arguments
parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()

# Initialize SwanLab
swanlab.init(project="sft", experiment_name="qwen-1.5b")

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Set the log level to INFO
transformers.utils.logging.set_verbosity_info()
log_level = training_args.get_process_log_level()
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()

# Record overall training status
logger.warning(
    f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
    + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
)
logger.info(f"Training/evaluation parameters {training_args}")

# Check for checkpoints
last_checkpoint = None
if os.path.isdir(training_args.output_dir):
    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
        raise ValueError(
            f"Output path ({training_args.output_dir}) is not empty "
        )
    elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(
            f"Resuming training from {last_checkpoint}"
        )

# Set the random seed.
set_seed(training_args.seed)

# Initialize the model
logger.warning("Loading the pretrained model")
logger.info(f"Model parameter path: {model_args.model_name_or_path}")
model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path,trust_remote_code=True)
n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
logger.info(f"Inheriting a pretrained model - Total size={n_params/2**20:.2f}M params")

# Initialize the Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
logger.info("Tokenizer loading complete")

# Load the fine-tuning data
with open(data_args.train_files) as f:
    lst = [json.loads(line) for line in f.readlines()[:10000]]
logger.info("Training set loading complete")
logger.info(f"Training set path: {data_args.train_files}")
logger.info(f'Total number of training samples: {len(lst)}')
# logger.info(f"Training set sample: {ds["train"][0]}")

train_dataset = SupervisedDataset(lst, tokenizer=tokenizer, max_len=2048)

logger.info("Initializing Trainer")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset= IterableWrapper(train_dataset),
    tokenizer=tokenizer
)

# Load from checkpoint
checkpoint = None
if training_args.resume_from_checkpoint is not None:
    checkpoint = training_args.resume_from_checkpoint
elif last_checkpoint is not None:
        checkpoint = last_checkpoint

logger.info("Starting training")
train_result = trainer.train(resume_from_checkpoint=checkpoint)
trainer.save_model() 
```

Launching also uses deepspeed in the sh script. We will not repeat it here. See ./code/finetune.sh for the source code.

## 6.3 Efficient Fine-Tuning

The previous sections introduced the principles and practical details of Pretrain, SFT, and RLHF on the Transformers framework. LLMs have many parameters and much training data. The methods above (mainly SFT and RLHF) train the model by adjusting all parameters. This work puts huge pressure on resources. Enterprises and research groups often have limited resources. They need a fast and cheap way to fine-tune the model for a domain or task. Then they can use LLMs to finish target tasks at low cost.

### 6.3.1 Efficient Fine-Tuning Approaches

Full fine-tuning is expensive. Two main solutions address this cost:

Adapter tuning: this method adds Adapter layers to the model. During fine-tuning, the original parameters stay frozen and only the Adapter layers are updated.

The method inserts downstream parameters into each layer of the pretrained model. These parameters are called Adapter modules. During fine-tuning, the model body stays frozen and only the task-specific parameters are trained. Figure 6.8 shows the idea.

<div align='center'>
    <img src="/images/6-images/3-1.webp" alt="Figure 6.8 Adapt Tuning" width="1444" height="1046" loading="lazy" decoding="async" />
    <p>Figure 6.8 Adapt Tuning</p>
</div>

Each Adapter module has two feed-forward sublayers. The first sublayer takes the output of the Transformer block as input. It projects the original input dimension $d$ down to $m$. The size of $m$ controls the number of Adapter parameters. Typically $m << d$. At the output stage, the second sublayer restores the dimension. It projects $m$ back to $d$ and gives the Adapter output. The right side of the figure above shows this structure.

LoRA is an improved Adapter tuning method. Adapter tuning has an inference latency problem. The extra parameters and extra computation make the fine-tuned model compute more slowly than the pretrained model.

Prefix tuning: this method freezes the pretrained LM and adds trainable, task-specific prefixes. You can save a different prefix for each task. The fine-tuning cost is also small. The method builds a segment of virtual tokens before each input token. These tokens relate to the downstream task. During fine-tuning, only the prefix parameters are updated. All other parameters stay frozen.

P-tuning is a common efficient fine-tuning method. It is an improvement on prefix tuning. Prefix tuning also has an inherent flaw. The virtual tokens take up available sequence length, so the model loses sequence length. High fine-tuning quality means low available sequence length.

### 6.3.2 LoRA Fine-Tuning

A large model maps data into a high-dimensional space. A narrow, small task does not need such a complex model. One subspace can solve it, so you do not need to optimize all parameters. Here is a definition. Suppose full-parameter optimization reaches a certain performance level, for example 90% accuracy. You can reach that level by optimizing only one subspace. The rank of that subspace's parameter matrix is then the intrinsic rank of the current problem.

Pretrained models themselves lower the intrinsic rank. After fine-tuning for a specific task, the weight matrices in the model have a lower intrinsic rank. A simpler downstream task gives a lower intrinsic rank. ([Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning](https://arxiv.org/abs/2012.13255)) The model can still learn the update parameter matrices. It can do this after a random projection into a smaller subspace. You can understand this fact this way. For specific downstream tasks, these weight matrices do not need full rank. You can train some dense layers in the neural network indirectly. Do this by optimizing the rank-decomposition matrices of how the dense layers change during adaptation. The fine-tuning effect then comes from optimizing only these rank-decomposition matrices.

For example, suppose the pretrained parameters are $\theta^D_0$. The intrinsic rank of the dense-layer weight matrix on a specific downstream task is $\theta^d$. The fine-tuning parameters for that task are $\theta^D$. Then:

$$\theta^D = \theta^D_0 + \theta^d M$$

This $M$ is the rank-decomposition matrix that LoRA optimizes.

Compared with other efficient fine-tuning methods, LoRA has the following advantages:

1. You can build small LoRA modules for different downstream tasks. You can then switch between tasks effectively and share pretrained model parameters.
2. LoRA uses an adaptive optimizer (Adaptive Optimizer). You do not need to compute gradients or keep optimizer states for most parameters. Training is more effective and needs less hardware.
3. LoRA uses a simple linear design. At deployment, the trainable matrices merge with the frozen weights. There is no inference latency.
4. LoRA is orthogonal to other methods and can combine with them.

LoRA is now the mainstream method for efficient LLM fine-tuning. It suits cases with limited resources and limited supervised training data. LoRA fine-tuning is often the first choice for LLM fine-tuning.

### 6.3.3 Principles of LoRA Fine-Tuning

#### (1) Low-Rank Parameterization of the Update Matrix

LoRA assumes that the weight update process also has a relatively low intrinsic rank. The pretrained weight matrix is $W0 \in R^{d \times k}$. Here $d$ is the output dimension of the previous layer and $k$ is the input dimension of the next layer. The method represents its update with a low-rank decomposition:

$$W_0 + {\Delta}W = W_0 + BA \space\space  where \space B \in R^{d \times r}, A \in R^{r \times k}$$

During training, $W_0$ stays frozen and does not update. $A$ and $B$ hold the trainable parameters.

LoRA's forward pass function is:

$$h = W_0 x + \Delta W x = W_0 x + B A x$$

At the start of training, $A$ uses random Gaussian initialization and $B$ uses zeros. The training then uses Adam for optimization.

Figure 6.9 shows the training idea:

<div align='center'>
    <img src="/images/6-images/3-2.webp" alt="Figure 6.9 LoRA" width="1057" height="852" loading="lazy" decoding="async" />
    <p>Figure 6.9 LoRA</p>
</div>

#### (2) Application to the Transformer

In the Transformer architecture, LoRA applies mainly to four weight matrices of the attention module: $W_q$, $W_k$, $W_v$, and $W_0$. The weight matrices of the MLP stay frozen.

Ablation experiments found that adjusting $W_q$ and $W_v$ together gives the best results.

Under these conditions, the number of trainable parameters is:

$$\Theta = 2 \times L_{LoRA} \times d_{model} \times r$$

Here $L_{LoRA}$ is the number of weight matrices that use LoRA. $d_{model}$ is the input/output dimension of the Transformer. $r$ is the configured LoRA rank.

In general, r is 4, 8, or 16.

### 6.3.4 LoRA Code Implementation

People now use the peft library to implement LoRA fine-tuning. Peft is a third-party library from HuggingFace. It collects many efficient fine-tuning methods, including LoRA, Adapter tuning, and P-tuning. You can use it to fine-tune models with LoRA easily.

This section parses the LoRA fine-tuning code in the peft library. It gives a simple analysis of the LoRA fine-tuning implementation.

#### (1) Implementation Flow

The LoRA fine-tuning implementation has these steps:

1. Find the layers that use LoRA. The peft library supports three layer types for LoRA: nn.Linear, nn.Embedding, and nn.Conv2d.

2. Replace each target layer with a LoRA layer. A LoRA layer adds a bypass on top of the layer's original result. The bypass simulates the parameter update through a low-rank decomposition, i.e., matrix $A$ and matrix $B$.

3. Freeze the original parameters. Run fine-tuning and update the LoRA layer parameters.

#### (2) Determining the LoRA Layers

LoRA fine-tuning first needs the LoRA fine-tuning parameters. One important parameter is target_modules. Target_modules is usually a list of strings. Each string is the name of a layer that uses LoRA, for example:

```python
target_modules = ["q_proj","v_proj"]
```

Here q_proj is $W_q$ in the attention mechanism and v_proj is $W_v$ in the attention mechanism. You can customize the LoRA layers to match the model architecture and the task requirements.

When the code creates the LoRA model, it reads this parameter and finds the matching layers in the original model. The code uses re to match layer names with a regex:

```python
# Find the components of the model whose names contain "q_proj", "v_proj"
target_module_found = re.fullmatch(self.peft_config.target_modules, key)
# Here key is the component name of the model
```

#### (3) Replacing with LoRA Layers

For each target layer that the code finds, it creates a new LoRA layer and replaces the old layer.

In its concrete implementation, the LoRA layer defines a Linear class based on the Lora base class. The class inherits from nn.Linear and LoraLayer. LoraLayer is the Lora base class. It builds LoRA's hyperparameters:

```python
class LoraLayer:
    def __init__(
        self,
        r: int, # The rank of LoRA
        lora_alpha: int, # Normalization parameter
        lora_dropout: float, # Dropout ratio of the LoRA layer
        merge_weights: bool, # In eval mode, whether to add the LoRA matrix values to the original weight matrix
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
        self.disable_adapters = False

```

nn.Linear is the PyTorch linear layer. The Linear class is the concrete LoRA layer. Its main implementation is:

```python
class Linear(nn.Linear, LoraLayer):
    # LoRA layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False, 
        merge_weights: bool = True,
        **kwargs,
    ):
        # Inherit the constructors of the two base classes
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoraLayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            # Parameter matrix A
            self.lora_A = nn.Linear(in_features, r, bias=False)
            # Parameter matrix B
            self.lora_B = nn.Linear(r, out_features, bias=False)
            # Normalization coefficient
            self.scaling = self.lora_alpha / self.r
            # Freeze the original parameters; only A and B are updated
            self.weight.requires_grad = False
        # Initialize A and B
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

```

During replacement, the code copies the weight and bias of the original layer into the new LoRA layer. It then assigns the new LoRA layer to the specified device.

#### (4) Training

After the LoRA layer replacement, fine-tuning training can start. The original parameters in the LoRA layer are already frozen, so training updates only the parameters of A and B. This update gives efficient fine-tuning. The overall training process is similar to the original fine-tune, so we will not repeat it here. The LoRA approach also changes the forward function:

```python
    def forward(self, x: torch.Tensor):
        if self.disable_adapters:
            if self.r > 0 and self.merged:
                self.weight.data -= (
                    transpose(self.lora_B.weight @ self.lora_A.weight, self.fan_in_fan_out) * self.scaling
                )
                self.merged = False

            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        '''Main branch'''
        elif self.r > 0 and not self.merged:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            if self.r > 0:
                result += self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
            return result
        else:
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

```

The code above has several branches because it handles parameter merging. Here you need to read only the second branch, i.e., the elif branch. The LoRA forward computation follows the formula above. The code first computes the product of the original parameters and the input. It then adds the products of A and B with the input.

### 6.3.5 LoRA Fine-Tuning with peft

peft is well encapsulated, so you can fine-tune large models easily and efficiently. Here we use the LLM SFT from Section 2 as an example. We briefly show how to use peft to fine-tune a large model. The approach is the same for RLHF.

First load the required libraries:

```python
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from transformers import Trainer
```

Next, load the original model and the original tokenizer. The method is the same as in Section 2:

```python
# Load the base model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModel.from_pretrained(
    MODEL_PATH, trust_remote_code=True
)
```

Then set the peft parameters:

```python
peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )
```

The LoRA parameters differ for different models. For ChatGLM you do not need to specify target_modules, because peft finds them on its own. For BaiChuan you must specify them manually. Task_type is the task type of the model. Large models are generally CAUSAL_LM, that is, traditional language models.

Then get the LoRA model:

```python
model = get_peft_model(model, peft_config)
```

The get_peft_model call here runs exactly the implementation analyzed above.

Finally, train with the Trainer from transformers. The GPU memory used for training will drop a lot:

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset= IterableWrapper(train_dataset),
    tokenizer=tokenizer
)
trainer.train()
```

DPO and KTO work the same way. Just add the LoRA parameters and get a LoRA model via `get_peft_model`. You need no other change. LoRA fine-tuning can greatly reduce GPU memory usage. It also gives good results on downstream task adaptation. But some tasks need new knowledge. LoRA adjusts only the low-rank matrices and finds it hard to inject knowledge, so results are generally poor. Do not use LoRA for model pretraining or post-training.

## 6.4 Preference Alignment

The preceding sections covered the three main training threads: Pretrain, SFT, and efficient fine-tuning. These steps are usually not enough for an LLM that serves real users. The model already has strong language modeling ability and basic instruction-following ability. It still needs more post-training to constrain and correct its answer style, safety, helpfulness, and stability. People call this step preference alignment.

### 6.4.1 Why Preference Alignment Is Needed

Pretrain focuses more on learning language patterns and world knowledge. SFT focuses more on teaching the model to follow instructions. Preference alignment emphasizes "which answers better match human expectations". Question answering, code generation, and complex reasoning tasks need more than an answer. The model must also try to express itself clearly, keep the content safe, keep the reasoning process stable, and avoid irrelevant answers or harmful output.

From the training pipeline view, preference alignment comes after SFT and is a key link in post-training. Its core idea is not to keep fitting text. It teaches the model to prefer the outputs that humans like more, among multiple candidate answers.

### 6.4.2 Overview of Mainstream Methods

Common preference alignment methods fall into these categories:

- RLHF: build preference data and a reward model first. Then update the policy model via reinforcement learning. This is the most classic preference alignment route.
- DPO: optimize the model directly with preference data. No reward model needs explicit training, so the training pipeline is relatively simpler.
- KTO: this method also belongs to direct preference optimization. It emphasizes a stable distinction between "good answers" and "bad answers".

These methods differ in details. They all rely on high-quality preference data and usually build on top of SFT models.

### 6.4.3 Connection with the Earlier Parts of This Chapter

From an engineering view, preference alignment is not a completely independent new system. It builds on the content of the previous sections of this chapter:

1. In `6.1`, we learned how to initialize models, process data, and organize the training pipeline on Transformers.
2. In `6.2`, we completed SFT data construction and supervised training. This step usually gives the starting model before preference alignment.
3. In `6.3`, we introduced LoRA/QLoRA and how to integrate with `peft`. You can also use these methods for preference alignment tasks such as DPO and KTO to reduce training costs.

When you practice preference alignment further, the new core content lies mainly in preference data organization, reward modeling, and the design of preference loss functions. Model loading, parameter-efficient training, and most engineering configurations can reuse the ideas from the earlier sections.

### 6.4.4 Learning and Practice Advice

If you learn LLM training systematically for the first time, master this chapter's Pretrain, SFT, and PEFT content fully first. Then read the preference alignment supplementary material [`6.4 Preference Alignment`](./6.4%5BWIP%5D%20偏好对齐.md). For practice, start with lighter preference optimization methods such as DPO first. Then build an understanding of the full chain of reward models and RLHF. In this way, you get an overall understanding of the post-training pipeline without a large increase in engineering complexity.

The sources for this chapter are listed below.

[1] Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin de Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. (2019). *Parameter-Efficient Transfer Learning for NLP.* arXiv preprint arXiv:1902.00751.

[2] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv preprint arXiv:2106.09685.

[3] Armen Aghajanyan, Luke Zettlemoyer, and Sonal Gupta. (2020). *Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning.* arXiv preprint arXiv:2012.13255.

[4] Xiang Lisa Li and Percy Liang. (2021). *Prefix-Tuning: Optimizing Continuous Prompts for Generation.* arXiv preprint arXiv:2101.00190.
