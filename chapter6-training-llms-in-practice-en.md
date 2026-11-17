# Chapter 6: LLM Training Pipeline in Practice

In Chapter 5, we built the LLaMA2 model entirely from scratch and implemented the full pipeline of its pretraining and fine-tuning. In this chapter, we take a deep dive into the practice of the LLM training pipeline, focusing on how to use mainstream LLM frameworks to train models and optimize performance efficiently.

> Reading advice: before starting the practice in this chapter, it is recommended to first read [Chapter 6 Practice Guide](./readme.md) and [Learning and Environment Setup](../学习与环境准备.md). The main text of this chapter focuses on three main threads: Pretrain, SFT, and PEFT. Preference alignment is treated as an advanced topic: Section 6.4 outlines the overall learning route and provides supplementary materials.

## 6.1 Model Pretraining

In the previous chapter, we dissected the model architecture and training process of LLMs step by step, hand-implementing the LLaMA model architecture and the full Pretrain and SFT pipelines from scratch, gaining a deeper understanding of LLM model principles and training details. However, in practical applications, hand-written LLM training has the following problems:

- Hand-implementing the LLM architecture requires a large amount of work, and it is difficult to keep up with the latest architectural innovations in real time;
- Training an LLM implemented from scratch cannot well support multi-GPU distributed training, and training efficiency is low;
- It is incompatible with existing pretrained LLMs, making it impossible to use pretrained model parameters.

Therefore, in this chapter, we introduce Transformers, the mainstream training framework in the LLM field, and, combined with mainstream frameworks such as the distributed framework DeepSpeed and the efficient fine-tuning framework peft, we practice the full Pretrain and SFT pipeline using transformers, to better align with the industry's mainstream LLM technical solutions.

### 6.1.1 Framework Introduction

Transformers is an NLP framework developed by Hugging Face. Through modular design, it provides unified support for hundreds of mainstream model architectures including BERT, GPT, LLaMA, T5, and ViT. With Transformers, developers no longer need to repeatedly implement basic network structures; any pretrained model can be loaded with a single click via the AutoModel class. Figure 6.1 shows the homepage of the Hugging Face Transformers course:

<div align='center'>
    <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/6-images/1-1.png" alt="alt text" width="90%">
    <p>Figure 6.1 Hugging Face Transformers</p>
</div>

Meanwhile, the Trainer class built into the framework encapsulates the core logic of distributed training and supports multiple distributed training strategies such as PyTorch native DDP, DeepSpeed, and Megatron-LM. By simply configuring the training parameters, you can achieve hybrid parallel training combining data parallelism, model parallelism, and pipeline parallelism, easily supporting efficient training of models with tens of billions of parameters on an 8-GPU A100 cluster. Together with components such as SavingPolicy and LoggingCallback, it enables automated management of the training process. It also supports integration with frameworks such as DeepSpeed, peft, wandb, and Swanlab, which can be seamlessly connected directly via parameter settings, enabling fast and efficient LLM training.

Even more important for NLP researchers in the LLM era, HuggingFace built its huge AI community on top of the Transformers framework, opening up hundreds of millions of pretrained model parameters and 250,000+ datasets of various types. Through multiple frameworks such as Transformers, Datasets, and Evaluate, it integrates pretrained models, datasets, and evaluation functions, helping developers conveniently use any pretrained model and easily develop and deploy their own models on the basis of open-source models and datasets.

<div align='center'>
    <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/6-images/1-2.png" alt="alt text" width="90%">
    <p>Figure 6.2 The Hugging Face Transformers model community</p>
</div>

In the LLM era, adjusting model architectures and re-pretraining are becoming less and less common; developers' business applications increasingly consist of using pretrained LLMs for Post-Training and SFT to support their downstream business applications. Moreover, because pretrained models are large in size, convenient integration with distributed training frameworks such as DeepSpeed has gradually become an essential skill for NLP model training in the LLM era. As a result, Transformers has gradually become the mainstream framework for NLP technology in both academia and industry; whether for enterprise business development or scientific research, Transformers is increasingly the first choice for model implementation. Meanwhile, newly released open-source LLMs such as DeepSeek and Qwen also open up their pretrained weights and model invocation demos in the Transformers community as soon as they are released. By using the Transformers framework, LLM training and development can be completed efficiently and conveniently, delivering industrial-grade output. Next, based on the Transformers framework, we will introduce how to implement LLM Pretrain and SFT with Transformers.

### 6.1.2 Initializing the LLM

We can use the AutoModel class from transformers to directly initialize an already-implemented model. For any pretrained model, its parameters contain the model's configuration information. If you want to train an LLM from scratch, you can initialize it directly using an existing model architecture. Here, we use the model architecture of [Qwen-2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B/tree/main) as an example:

<div align='center'>
    <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/6-images/1-3.png" alt="alt text" width="90%">
    <p>Figure 6.3 Qwen-2.5-1.5B</p>
</div>

This page shows the Qwen-2.5-1.5B model parameters in the HuggingFace community. The `config.json` file contains the model's configuration information, including the model architecture, hidden size, number of layers, etc., as shown in Figure 6.4:

<div align='center'>
    <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/6-images/1-4.png" alt="alt text" width="90%">
    <p>Figure 6.4 The Qwen-2.5-1.5B config.json file</p>
</div>

We can reuse this model's configuration information to initialize a Qwen-2.5-1.5B model for training, or modify the configuration information—e.g., changing the hidden size or the number of attention heads—to customize a model architecture. HuggingFace provides Python tools to conveniently download the model parameters you want to use:

```python
import os
# Set the environment variable; here we use the HuggingFace mirror site
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# Download the model
os.system('huggingface-cli download --resume-download Qwen/Qwen2.5-1.5B --local-dir your_local_dir')
```

As shown in Figure 6.5, "Qwen/Qwen2.5-1.5B" here is the identifier of the model to download; for other models, you can directly copy the model name from HuggingFace:

<div align='center'>
    <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/6-images/1-5.png" alt="alt text" width="90%">
    <p>Figure 6.5 Model download identifier</p>
</div>

After downloading, you can use the AutoConfig class to directly load the downloaded configuration file:

```python
# Load the predefined model parameters - here using Qwen-2.5-1.5B as an example
# Load using the Config class from transformers
from transformers import AutoConfig

# Local path of the downloaded parameters
model_path = "qwen-1.5b"
config = AutoConfig.from_pretrained(model_name_or_path)
```

You can also customize the configuration file and then load it in the same way. You can use the AutoModel class to generate the corresponding model from the loaded configuration object:

```python
# Generate a defined model from this configuration
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_config(config,trust_remote_code=True)
```

Since LLMs generally use the CausalLM architecture, we use the AutoModelForCausalLM class for loading here. For classification-task training, you can use the AutoModelForSequenceClassification class instead. Inspecting this model, Figure 6.6 shows that its architecture matches the defined configuration file:

<div align='center'>
    <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/6-images/1-6.png" alt="alt text" width="70%">
    <p>Figure 6.6 Model architecture output</p>
</div>

This model is a Qwen-2.5-1.5B model initialized from scratch. In general, we rarely pretrain an LLM initialized from scratch; a more common practice is to load a pretrained LLM's weights and continue training on your own corpus. Here, we also introduce how to initialize a pretrained model from downloaded model parameters.

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(model_name_or_path,trust_remote_code=True)
```

Similarly, just load directly with the from_pretrained method; model_name_or_path here is the local path of the downloaded parameters.

We also need to initialize a tokenizer. Here, we can directly use the tokenizer parameters corresponding to Qwen-2.5-1.5B:

```python
# Load a pretrained tokenizer
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
```

The loaded tokenizer can be used directly to tokenize any text.

### 6.1.3 Pretraining Data Processing

As in Chapter 5, we use the Mobvoi Seq Monkey open-source dataset as the pretraining dataset, and the dataset can be downloaded and extracted in the same way as in Chapter 5. HuggingFace's datasets library is a third-party library that pairs with the transformers framework for data downloading and processing. We can directly use the load_dataset function of datasets to load the pretraining data:

```python
# Load the pretraining data
from datasets import load_dataset

ds = load_dataset('json', data_files='/mobvoi_seq_monkey_general_open_corpus.jsonl')
```

Note that since the dataset is large, loading may take a long time or run out of memory; it is recommended to split off a portion of the pretraining dataset for early testing. The loaded ds is a DatasetDict object, and the loaded data is saved by default under the `train` key, which can be inspected with the following code:

```python
ds["train"][0]
```

<div align='center'>
    <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/6-images/1-7.png" alt="alt text" width="100%">
    <p>Figure 6.7 Dataset preview</p>
</div>

You can view the dataset's features (i.e., its columns) through the features attribute. Here we need to save the dataset's column names, because in subsequent data processing, after tokenizing the text, the original text needs to be removed:

```python
# View the features
column_names = list(ds["train"].features)
# columnes_name:["text"]
```

Next, use the loaded tokenizer to process the dataset, using the map function for batch processing:

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

The processed dataset contains two columns, 'input_ids' and 'attention_mask', which are the tokenized numeric sequences of the text and the attention mask (indicating whether padding was applied). The map method removes the original 'text' column via the remove_columns parameter, and it is no longer used in training.

Since pretraining is generally a CLM task, learning the sequence semantics of multiple samples at once does not affect model performance; and because the training data is large and training takes long, the requirements on training efficiency are high. During pretraining, multiple text segments are usually concatenated together and processed into uniformly sized text blocks, and each text block is then used for training. Here, we implement a concatenation function that concatenates text into blocks of 2048 tokens, and use the map method for batch processing:

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

The resulting train_dataset is a pretraining dataset that can be used directly for CLM Pretrain, with each sample 2048 tokens long.

### 6.1.4 Training with the Trainer

Next, we use the Trainer class provided by transformers for training. Trainer encapsulates the model training logic and includes good efficiency optimizations, visualization, and more, enabling efficient and convenient LLM training.

First we need to configure the training hyperparameters, using the TrainingArguments class to instantiate a parameter object:

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

Then, based on the initialized model, tokenizer, and training_args, and passing in the processed training dataset, instantiate a trainer object:

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

Then use the train method, which will train and save according to the configured training hyperparameters:

```python
trainer.train()
```

> Note: the above code is stored in the `./code/pretrain.ipynb` file.

### 6.1.5 Distributed Training with DeepSpeed

Because pretraining is large in scale and long in duration, running it in a Jupyter Notebook is generally not recommended, as it is prone to interruption. Also, due to the large scale of pretraining, multi-GPU distributed training is usually required; otherwise training takes too long. Here, we introduce how, based on the above code, to use the DeepSpeed framework for distributed training, in order to accomplish industry-usable LLM Pretrain.

Long-running training generally uses a bash script to set the hyperparameters and then launches a pre-written Python script to run the training. We use a Python script (`./code/pretrain.py`) to implement the full training pipeline.

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

First, we need to define several hyperparameter classes to handle the hyperparameter values set in the sh script. Since transformers already has the TrainingArguments class, which includes the essential training hyperparameters, we only need to define the hyperparameters not included in TrainingArguments, mainly the model-related hyperparameters (defined in ModelArguments) and the data-related hyperparameters (defined in DataTrainingArguments):

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

Then we can define a main function to wrap the training process described above. First, use the HfArgumentParser tool provided by transformers to load the hyperparameters set in the sh script:

```python
# Load script arguments
parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()
```

In large-scale training, logs are generally used to record information about the training process; directly printing with print is not recommended, as key training information can be lost. Here, we directly use Python's built-in logging library for logging. First, logging needs to be configured:

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

Here the log level is set to INFO. The logging library has five levels in total: DEBUG, INFO, WARNING, ERROR, and CRITICAL. Whichever level you set, only messages at that level and above will be output. Once configured, use logger directly wherever logging is needed, specifying the log level when recording, for example:

```python
# Record overall training status
logger.warning(
    f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
    + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
)
logger.info(f"Training/evaluation parameters {training_args}")
```

We will not go into further detail about logging in the script below.

In large-scale training, interruptions are often unavoidable. Training generally saves checkpoints at fixed intervals, and after an interruption you can simply resume training from the most recent checkpoint. Therefore, we first need to check whether an old checkpoint exists and resume training from the checkpoint:

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

Next, initialize the model in the way introduced above; here, from-scratch initialization and initialization from an existing pretrained model are wrapped together:

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

Then, similarly, load the tokenizer and process the pretraining data. This part is exactly the same as above, so we will not repeat it here; readers can check the details in the code. Similarly, use the Trainer for training:

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

Note that since we checked for existing checkpoints above, resume_from_checkpoint is used here to implement the functionality of resuming training from a checkpoint.

Because monitoring training progress and the downward trend of the loss is especially important in large-scale training, the script uses swanlab as the training monitoring tool. swanlab is initialized at the beginning of the script:

```python
# Initialize SwanLab
swanlab.init(project="pretrain", experiment_name="from_scrach")
```

After training starts, the terminal prints the swanlab monitoring URL, which you can click to observe training progress. We will not go into the details of using swanlab here; readers are welcome to consult the relevant documentation.

After completing the above code, we use a sh script (`./code/pretrain.sh`) to define the hyperparameter values and launch the training through DeepSpeed, achieving efficient multi-GPU distributed training:

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

After installing the DeepSpeed third-party library, you can launch multi-GPU training directly with the deepspeed command. The above script commands mainly define the values of the various hyperparameters and can be used as a reference. In Chapter 4, we introduced the principles of DeepSpeed distributed training and the ZeRO stage settings; here, we train with ZeRO-2. Here, `ds_config_zero.json` is loaded as the DeepSpeed configuration parameters:

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

In the previous section, we introduced how to use the Transformers framework to perform model pretraining quickly and efficiently. In this part, building on the previous content, we introduce how to use the Transformers framework to perform supervised fine-tuning on a pretrained model.

### 6.2.1 Pretrain VS SFT

First, we need to review what exactly the core difference is between pretraining an LLM and supervised fine-tuning it. As mentioned in Chapter 4, current mature LLMs are generally trained through the three stages Pretrain-SFT-RLHF. In the Pretrain stage, massive unlabeled text is modeled in a self-supervised manner to learn text semantic rules and the world knowledge contained in the text; in the SFT stage, the pretrained model is generally instruction-tuned, i.e., the model is trained to complete corresponding tasks according to user instructions, so that the model can follow user instructions and plan, act, and produce output based on them. Therefore, both Pretrain and SFT use CLM modeling, and the core difference is that Pretrain trains on massive unlabeled text, where the model directly performs the "predict the next token" task on the text, while SFT uses constructed paired instruction data, where the model models the subsequent output given the input instruction. Reflected in the concrete training implementation, Pretrain computes loss over the entire text, requiring the model to model and predict the whole text; whereas SFT computes loss only on the output, without computing loss on the instruction part.

Therefore, compared with the Pretrain code completed in the previous section, the SFT part only needs to modify the data processing step, implementing the construction that converts paired instruction data into training samples; the rest follows exactly the same implementation logic as Pretrain. The code script for this part is `./code/finetune.py`.

### 6.2.2 Fine-tuning Data Processing

Similarly, as in Chapter 5, we use the BelleGroup dataset open-sourced by Beike for SFT here.

During SFT, we define a Chat Template, which specifies how to convert conversation data into a text sequence that the model can model and fit. When we use a model that has undergone SFT for downstream task fine-tuning, we generally need to check that model's Chat Template and adapt to it, in order not to damage the instruction-following ability it learned during SFT. Since we use a Pretrain model for SFT here, we can define a custom Chat Template. Because we used the Qwen-2.5-1.5B model architecture for Pretrain, we continue to use the Qwen-2.5 Chat Template here. If readers do not have enough resources to pretrain the model as in the previous part, they can also use the official Qwen-2.5-1.5B model as the base model for SFT.

We first define several special tokens. Special tokens play special roles during model fitting, including the beginning of the text sequence (BOS), the end of the text sequence (EOS), newline characters, etc. Defining special tokens helps avoid semantic confusion during model fitting:

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

The Chat Template of the Qwen series generally has three conversation roles: System, User, and Assistant. System is the system prompt, responsible for activating the model's capabilities; the default is "You are a helpful assistant.", and it is generally not changed during SFT. User is the prompt given by the user; here, since the conversation role in the dataset is "human", we changed "user" to "human". Assistant is the reply given by the LLM, i.e., the text the model needs to fit during SFT.

Next, since this dataset is a multi-turn conversation dataset, we need to concatenate the multi-turn conversations, joining the multiple turns into a single text sequence:

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

The above code follows the Qwen Chat Template logic, which readers can modify according to their own preferences. Its key point is that the User's text does not need to be fitted, so the User's corresponding text content in targets is masked using IGNORE_TOKEN_ID, while the Assistant's corresponding text content is the original text, for which loss is computed. Currently, for mainstream LLMs, IGNORE_TOKEN_ID is generally set to -100.

After concatenation, convert the tokenized numeric sequences into `Torch.tensor`, and then assemble them into the dictionary required by the Dataset to return:

```python
input_ids = torch.tensor(input_ids)
targets = torch.tensor(targets)

return dict(
    input_ids=input_ids,
    labels=targets,
    attention_mask=input_ids.ne(tokenizer.pad_token_id),
)
```

After completing the above processing logic, you need to define a custom Dataset class that calls this logic to process the data:

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

This class inherits from Torch's Dataset class and can be used directly in the Trainer. After the data processing is done, based on the previous section's script, you only need to modify the data processing logic; the subsequent model training and so on is almost entirely the same. The main function logic is attached here:

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

Launching is likewise done with deepspeed in the sh script; we will not repeat it here. See ./code/finetune.sh for the source code.

## 6.3 Efficient Fine-Tuning

In the previous sections, we introduced in detail the principles and practical details of Pretrain, SFT, and RLHF based on the Transformers framework. However, because LLMs have a large number of parameters and a lot of training data, training the model via the above methods (mainly SFT and RLHF) requires adjusting all of the model's parameters, which puts enormous pressure on resources. For enterprises or research groups with limited resources, how to efficiently and quickly fine-tune the model for a domain or task, so as to use LLMs to accomplish target tasks at low cost, is very important.

### 6.3.1 Efficient Fine-Tuning Approaches

To address the cost of full fine-tuning, there are currently two main solutions:

**Adapt Tuning**: i.e., adding Adapter layers to the model; during fine-tuning, the original parameters are frozen and only the Adapter layers are updated.

Specifically, it inserts parameters for downstream tasks, i.e., Adapter modules, into each layer of the pretrained model; during fine-tuning, the model body is frozen and only the task-specific parameters are trained, as shown in Figure 6.8.

<div align='center'>
    <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/6-images/3-1.png" alt="alt text" width="90%">
    <p>Figure 6.8 Adapt Tuning</p>
</div>

Each Adapter module consists of two feed-forward sublayers. The first feed-forward sublayer takes the output of the Transformer block as its input and projects the original input dimension $d$ down to $m$, controlling the number of parameters of the Adapter module by controlling the size of $m$; typically $m << d$. At the output stage, the second feed-forward sublayer restores the input dimension, projecting $m$ back to $d$, as the output of the Adapter module (the structure on the right in the figure above).

LoRA is in fact an improved Adapt Tuning method. However, the Adapt Tuning approach has an inference latency problem: because of the added extra parameters and extra computation, the fine-tuned model computes more slowly than the original pretrained model.

**Prefix Tuning**: this method freezes the pretrained LM and adds trainable, task-specific prefixes to the LM, so that different prefixes can be saved for different tasks, and the fine-tuning cost is also small. Specifically, a segment of virtual tokens related to the downstream task is constructed before each input token as the prefix; during fine-tuning, only the parameters of the prefix part are updated, while the other parameters remain frozen.

P-tuning, a currently commonly used efficient fine-tuning method, is in fact an improvement of Prefix Tuning. But Prefix Tuning also has an inherent flaw: the model's available sequence length is reduced. Because the virtual tokens take up the available sequence length, the higher the fine-tuning quality, the lower the model's available sequence length.

### 6.3.2 LoRA Fine-Tuning

If a large model maps data into a high-dimensional space for processing, then when handling a narrow small task, such a complex large model may not be needed; it may be solvable within a certain subspace, so there is no need to optimize all parameters. We can define: when optimizing the parameters of a certain subspace can reach a certain level of the performance of full-parameter optimization (e.g., 90% accuracy), the rank of that subspace's parameter matrix can be called the intrinsic rank of the current problem to be solved.

Pretrained models themselves implicitly reduce the intrinsic rank; after fine-tuning for a specific task, the weight matrices in the model actually have a lower intrinsic rank. Meanwhile, the simpler the downstream task, the lower the corresponding intrinsic rank. ([Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning](https://arxiv.org/abs/2012.13255)) Therefore, the parameter matrices of the weight updates can still be learned effectively even when randomly projected into a smaller subspace; it can be understood that for specific downstream tasks, these weight matrices do not need to be full rank. We can indirectly train some dense layers in the neural network by optimizing the rank-decomposition matrices of how the dense layers change during adaptation, thereby achieving the fine-tuning effect by only optimizing the rank-decomposition matrices of the dense layers.

For example, suppose the pretrained parameters are $\theta^D_0$, the intrinsic rank corresponding to the dense-layer weight parameter matrix on a specific downstream task is $\theta^d$, and the fine-tuning parameters for the specific downstream task are $\theta^D$; then:

$$\theta^D = \theta^D_0 + \theta^d M$$

This $M$ is the rank-decomposition matrix optimized by LoRA.

Compared with other efficient fine-tuning methods, LoRA has the following advantages:

1. Small LoRA modules can be built for different downstream tasks, enabling effective switching between downstream tasks on the basis of shared pretrained model parameters.
2. LoRA uses an adaptive optimizer (Adaptive Optimizer); there is no need to compute gradients or maintain optimizer states for most parameters, making training more effective and lowering the hardware threshold.
3. LoRA uses a simple linear design; at deployment, the trainable matrices are merged with the frozen weights, so there is no inference latency.
4. LoRA is orthogonal to other methods and can be combined with them.

Therefore, LoRA has become the mainstream method for efficient LLM fine-tuning, especially in cases with limited resources and limited supervised training data, where LoRA fine-tuning is often the first choice for LLM fine-tuning.

### 6.3.3 Principles of LoRA Fine-Tuning

#### (1) Low-Rank Parameterization of the Update Matrix

LoRA assumes that the weight update process also has a relatively low intrinsic rank. For the pretrained weight parameter matrix $W0 \in R^{d \times k}$ ($d$ is the output dimension of the previous layer, $k$ is the input dimension of the next layer), its update is represented using a low-rank decomposition:

$$W_0 + {\Delta}W = W_0 + BA \space\space  where \space B \in R^{d \times r}, A \in R^{r \times k}$$

During training, $W_0$ is frozen and not updated; $A$ and $B$ contain the trainable parameters.

Therefore, LoRA's forward pass function is:

$$h = W_0 x + \Delta W x = W_0 x + B A x$$

At the start of training, $A$ is initialized with random Gaussian initialization and $B$ is initialized with zeros, and then Adam is used for optimization.

The training idea is shown in Figure 6.9:

<div align='center'>
    <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/6-images/3-2.jpg" alt="alt text" width="90%">
    <p>Figure 6.9 LoRA</p>
</div>

#### (2) Application to the Transformer

In the Transformer architecture, the LoRA technique is mainly applied to the four weight matrices of the attention module: $W_q$, $W_k$, $W_v$, and $W_0$, while the weight matrices of the MLP are frozen.

Ablation experiments found that adjusting $W_q$ and $W_v$ simultaneously produces the best results.

Under the above conditions, the number of trainable parameters is:

$$\Theta = 2 \times L_{LoRA} \times d_{model} \times r$$

where $L_{LoRA}$ is the number of weight matrices to which LoRA is applied, $d_{model}$ is the input/output dimension of the Transformer, and $r$ is the configured LoRA rank.

In general, r is chosen as 4, 8, or 16.

### 6.3.4 LoRA Code Implementation

Currently, the peft library is generally used to implement LoRA fine-tuning of models. peft is a third-party library developed by HuggingFace that encapsulates a variety of efficient fine-tuning methods including LoRA, Adapt Tuning, and P-tuning, allowing convenient LoRA fine-tuning of models based on it.

This article briefly parses the LoRA fine-tuning code in the peft library, giving a simple analysis of the code implementation of LoRA fine-tuning.

#### (1) Implementation Flow

The internal implementation flow of LoRA fine-tuning mainly includes the following steps:

1. Determine the layers to which LoRA is to be applied. The layers currently supported by the peft library for applying LoRA include three types: nn.Linear, nn.Embedding, and nn.Conv2d.

2. For each layer to which LoRA is to be applied, replace it with a LoRA layer. A so-called LoRA layer is, in fact, a bypass added on top of the layer's original result, simulating parameter updates through low-rank decomposition (i.e., matrix $A$ and matrix $B$).

3. Freeze the original parameters, perform fine-tuning, and update the LoRA layer parameters.

#### (2) Determining the LoRA Layers

When performing LoRA fine-tuning, you first need to determine the LoRA fine-tuning parameters; one important parameter is target_modules. target_modules is generally a list of strings, where each string is the name of a layer to which LoRA should be applied, for example:

```python
target_modules = ["q_proj","v_proj"]
```

Here q_proj is $W_q$ in the attention mechanism, and v_proj is $W_v$ in the attention mechanism. We can customize the layers to which LoRA is applied according to the model architecture and task requirements.

When creating the LoRA model, this parameter is fetched and the corresponding layers are then found in the original model; this operation is mainly implemented by using re to regex-match the layer names:

```python
# Find the components of the model whose names contain "q_proj", "v_proj"
target_module_found = re.fullmatch(self.peft_config.target_modules, key)
# Here key is the component name of the model
```

#### (3) Replacing with LoRA Layers

For each found target layer, a new LoRA layer is created to replace it.

In its concrete implementation, the LoRA layer defines a Linear class based on the Lora base class, which inherits from both nn.Linear and LoraLayer. LoraLayer is the Lora base class, which mainly constructs LoRA's various hyperparameters:

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

nn.Linear is PyTorch's linear layer implementation. The Linear class is the concrete LoRA layer, implemented mainly as follows:

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

When replacing, the weight and bias of the original layer are directly copied to the new LoRA layer, and then the new LoRA layer is assigned to the specified device.

#### (4) Training

After the LoRA layer replacement is implemented, fine-tuning training can proceed. Since the original parameters are already frozen in the LoRA layer, only the parameters of A and B are updated during training, thereby achieving efficient fine-tuning. The overall training process is similar to the original Fine-tune, so we will not repeat it here. Because the LoRA approach is adopted, the forward function is also adjusted accordingly:

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

The above code has several branches because it takes parameter merging into account; here we only need to read the second branch, i.e., the elif branch. The LoRA forward computation process is as shown in the formula above: first compute the product of the original parameters with the input, then add the products of A and B with the input respectively.

### 6.3.5 LoRA Fine-Tuning with peft

peft is well encapsulated, enabling us to fine-tune large models conveniently and efficiently. Here, using the LLM SFT from Section 2 as an example, we briefly introduce how to use peft to fine-tune a large model. If applied to RLHF, the overall approach is the same.

First load the required libraries:

```python
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from transformers import Trainer
```

Next, load the original model and the original tokenizer, in the same way as in Section 2:

```python
# Load the base model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModel.from_pretrained(
    MODEL_PATH, trust_remote_code=True
)
```

Then, set the peft parameters:

```python
peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )
```

Note that the LoRA parameters may differ for different models. For example, for ChatGLM there is no need to specify target_modeules, as peft can find them on its own; for BaiChuan, you need to specify them manually. task_type is the model's task type; large models are generally CAUSAL_LM, i.e., traditional language models.

Then, obtain the LoRA model:

```python
model = get_peft_model(model, peft_config)
```

The underlying operation of get_peft_model here is exactly the concrete implementation analyzed above.

Finally, train with the Trainer provided by transformers; the GPU memory used for training will drop substantially:

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset= IterableWrapper(train_dataset),
    tokenizer=tokenizer
)
trainer.train()
```

If applied to DPO or KTO, likewise just add the LoRA parameters and obtain a LoRA model via `get_peft_model`; nothing else needs to be modified. However, note that while LoRA fine-tuning can greatly reduce GPU memory usage and achieves good results on downstream task adaptation, for tasks that require learning corresponding knowledge, LoRA only adjusts the low-rank matrices and has difficulty injecting knowledge, so the results are generally poor; therefore, LoRA is not recommended for model pretraining or post-training.

## 6.4 Preference Alignment

In the preceding sections, we have completed the three main training threads: Pretrain, SFT, and efficient fine-tuning. For an LLM serving real users, these steps are usually not enough. Although the model already has strong language modeling ability and basic instruction-following ability, further post-training is still needed to constrain and correct its answer style, safety, helpfulness, and stability; this step is generally collectively referred to as preference alignment.

### 6.4.1 Why Preference Alignment Is Needed

Pretrain focuses more on learning language patterns and world knowledge; SFT focuses more on teaching the model to follow instructions; while preference alignment emphasizes "which answers better match human expectations". For example, in question answering, code generation, or complex reasoning tasks, the model must not only give an answer, but also try to express itself clearly, keep the content safe, keep the reasoning process stable, and avoid irrelevant answers or harmful output as much as possible.

From the perspective of the training pipeline, preference alignment comes after SFT and is a key link in Post-Training. Its core idea is not to have the model continue simply fitting text, but to teach the model to prefer, among multiple candidate answers, the outputs that humans like more.

### 6.4.2 Overview of Mainstream Methods

Currently common preference alignment methods mainly include the following categories:

- **RLHF**: first build preference data and a reward model, then update the policy model via reinforcement learning; it is the most classic preference alignment route.
- **DPO**: directly optimizes the model using preference data, without explicitly training a reward model; the training pipeline is relatively simpler.
- **KTO**: also belongs to direct preference optimization methods, placing more emphasis on making the model form a stable distinction between "good answers" and "bad answers".

Although these methods differ in details, they all rely on high-quality preference data, and they are usually built on top of SFT models.

### 6.4.3 Connection with the Earlier Parts of This Chapter

From an engineering implementation perspective, preference alignment is not a completely independent new system; rather, it is built on top of the content of the previous sections of this chapter:

1. In `6.1`, we have already learned how to initialize models, process data, and organize the training pipeline based on Transformers.
2. In `6.2`, we have completed SFT data construction and supervised training; this step is usually the starting model before preference alignment.
3. In `6.3`, we introduced LoRA/QLoRA and how to integrate with `peft`; these methods can likewise be used for preference alignment tasks such as DPO and KTO to reduce training costs.

Therefore, when you further practice preference alignment, the truly new core content mainly lies in preference data organization, reward modeling, and the design of preference loss functions, while model loading, parameter-efficient training, and most engineering configurations can reuse the ideas from the earlier sections.

### 6.4.4 Learning and Practice Advice

If you are systematically learning LLM training for the first time, it is recommended to first fully master this chapter's Pretrain, SFT, and PEFT content, and then continue reading the preference alignment supplementary material [`6.4 Preference Alignment`](./6.4%5BWIP%5D%20偏好对齐.md). In terms of practice, it is recommended to start first with lighter-weight preference optimization methods such as DPO, and then gradually understand the complete chain of reward models and RLHF. In this way, you can build an overall understanding of the post-training pipeline without significantly increasing engineering complexity.

**References**

[1] Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin de Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. (2019). *Parameter-Efficient Transfer Learning for NLP.* arXiv preprint arXiv:1902.00751.

[2] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv preprint arXiv:2106.09685.

[3] Armen Aghajanyan, Luke Zettlemoyer, and Sonal Gupta. (2020). *Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning.* arXiv preprint arXiv:2012.13255.

[4] Xiang Lisa Li and Percy Liang. (2021). *Prefix-Tuning: Optimizing Continuous Prompts for Generation.* arXiv preprint arXiv:2101.00190.
