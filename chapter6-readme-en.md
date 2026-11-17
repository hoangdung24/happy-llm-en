# Chapter 6 Practice Guide

Chapter 6 focuses on how to complete LLM training practice based on the Transformers ecosystem. The main text follows three main threads—Pretrain, SFT, and PEFT—and serves as a bridge chapter from hand-written model implementations to industrial training frameworks.

## 1. What This Chapter Contains

- [`Chapter 6: LLM Training Pipeline in Practice`](./第六章%20大模型训练流程实践.md): the main text of Chapter 6, introducing core content such as the training framework, data processing, Trainer, DeepSpeed, and LoRA/QLoRA.
- [`6.4 Preference Alignment Supplementary Topic`](./6.4%5BWIP%5D%20偏好对齐.md): supplementary reading material on preference alignment, to be studied after completing the main text.
- [`code/`](./code): the code directory accompanying this chapter, containing examples for data download, pretraining, SFT, and DeepSpeed configuration.

## 2. Recommended Learning Order

1. Read `6.1 Model Pretraining` in the main text to understand the basic usage of Transformers, Trainer, and DeepSpeed.
2. Read `6.2 Supervised Fine-Tuning` to compare the differences in data construction and loss computation between Pretrain and SFT.
3. Read `6.3 Efficient Fine-Tuning` to grasp the basic ideas of LoRA/QLoRA and how to integrate with `peft`.
4. After completing the above, read the preference alignment supplementary material to build a complete picture from SFT to Post-Training.

## 3. Code Entry Points

The main code files of this chapter are as follows:

- [`code/download_model.py`](./code/download_model.py): download the base model.
- [`code/download_dataset.py`](./code/download_dataset.py): download or prepare the training data.
- [`code/pretrain.py`](./code/pretrain.py): the pretraining script.
- [`code/finetune.py`](./code/finetune.py): the supervised fine-tuning script.
- [`code/pretrain.sh`](./code/pretrain.sh): a DeepSpeed pretraining launch example.
- [`code/finetune.sh`](./code/finetune.sh): a DeepSpeed fine-tuning launch example.
- [`code/ds_config_zero2.json`](./code/ds_config_zero2.json): the DeepSpeed configuration file.

## 4. Environment Suggestions

- Dependency file: [`code/requirements.txt`](./code/requirements.txt)
- Recommended Python version: 3.10 or 3.11
- Recommended hardware: multi-GPU; if resources are limited, it is recommended to first use small samples or a single-GPU environment to debug the data processing and training pipeline

You can refer to the [`Learning and Environment Setup`](../学习与环境准备.md) page in the repository to prepare the environment in a unified way.

## 5. Practice Suggestions

### 5.1 Get the Pipeline Working First, Then Scale Up

The `autodl-tmp`, GPU indices, batch size, and output directories in this chapter's scripts are all example configurations. It is recommended to first replace the model paths, data paths, and output paths with your actual local paths, and then use a small sample to verify that the pipeline runs end to end.

### 5.2 Understand the Data First, Then Start Training

Whether Pretrain or SFT, the data format directly determines the training results. It is recommended to first read the data processing part of the main text, and to print 1 to 2 samples for inspection before running.

### 5.3 Treat Preference Alignment as an Advanced Topic

The main text of Chapter 6 already covers the core training threads. If you want to further understand RLHF, DPO, KTO, and reward models, you can continue reading the preference alignment supplementary material after completing the main text, rather than having to master all post-training details in a single first pass.
