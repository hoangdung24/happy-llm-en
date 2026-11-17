---
title: "Chapter 4: Large Language Models"
---

## 4.1 What is an LLM

In the previous three chapters, starting from the definition and main tasks of NLP, we introduced the core ideas that triggered the major transformation of the NLP field — the attention mechanism and the Transformer architecture. With the debut of the Transformer architecture, the NLP field gradually entered the pre-training–fine-tuning paradigm. Pre-trained language models based on the Transformer, which acquire powerful text representation capabilities through pre-training, emerged in great numbers, pushing all kinds of classic NLP tasks to a new height.

At the end of 2022, ChatGPT once again raised the capability ceiling of NLP, and Large Language Models (LLMs) began to replace traditional Pre-trained Language Models (PLMs) as the mainstream direction of NLP. The brand-new research paradigm based on LLMs is also refreshing the pre-training–fine-tuning paradigm that BERT made famous, and NLP thus welcomes another earth-shaking transformation. From the end of 2022 until now, the capability ceiling of LLMs has been continuously pushed upward, the number of general-purpose foundation models has grown exponentially, and LLM-based concepts and applications are changing with each passing day, heralding the arrival of the era of large models.

In Chapter 3, starting from the perspective of model architecture, we respectively analyzed the classic models under the three architectures — Encoder-Only, Encoder-Decoder, and Decoder-Only — and their training processes. Some of these models are milestones that can be regarded as the protagonists of their era before the LLM age (such as BERT), while others are the protagonists of the LLM era's stage and strong contenders for Artificial General Intelligence (AGI). So, what exactly is an LLM? Where do the core differences between LLMs and traditional PLMs lie? And what makes researchers hold such high enthusiasm and expectations for LLMs?

In this chapter, building on the model architecture explanations above, we will deeply analyze the definition, characteristics, and capabilities of LLMs, revealing to readers the core differences between LLMs and traditional deep learning models. On this basis, we will present the actual three-stage training process of LLMs, helping readers conceptually clarify how LLMs acquire such unique capabilities, thereby providing the theoretical foundation for the further practice of full LLM training.

### 4.1.1 The Definition of an LLM

An LLM, i.e., a Large Language Model, known in Chinese as 大语言模型 or 大型语言模型 (both meaning "large language model"), is a language model with far more parameters than traditional language models, pre-trained on much larger-scale corpora.

In Chapter 1, we already introduced the concept of a language model, i.e., an NLP model trained by predicting the next token. LLMs use architectures and pre-training tasks similar to those of traditional pre-trained language models (such as the Decoder-Only architecture and the CLM pre-training task), but they possess much larger parameter counts and are pre-trained on vastly more massive corpora, and thus exhibit capabilities strikingly different from those of traditional pre-trained language models.

Generally speaking, an LLM refers to a **language model with tens of billions (or more) of parameters**, often pre-trained **on several T tokens of corpus** via multi-GPU distributed clusters, possessing text understanding and generation capabilities far beyond those of traditional pre-trained models. However, as LLM research continues to deepen, LLMs of various parameter sizes have gradually become abundant; in the broad sense, an LLM generally covers all large language models from **one billion parameters** (e.g., Qwen-1.5B) to **hundreds of billions of parameters** (e.g., Grok-3 at 14B). As long as a model exhibits **emergent abilities**, i.e., capabilities and potential on a series of complex tasks that far exceed those of traditional pre-trained models (such as BERT and T5), it can be called an LLM.

It is generally believed that GPT-3 (175 billion parameters) marked the beginning of LLMs. ChatGPT, obtained from GPT-3 through three-stage training — Pretraining, Supervised Fine-Tuning (SFT), and Reinforcement Learning with Human Feedback (RLHF) — dominated the arrival of the LLM era. In less than 2 years since OpenAI released ChatGPT in November 2022, hundreds of LLMs with distinct characteristics and varying abilities have emerged. The table below lists some of the large models released at home and abroad from November 2022 to November 2023:

Time     | Open-source LLMs                                       | Closed-source LLMs
-------- | -----                                         | --------
2022.11  | None                                            | OpenAI-ChatGPT
2023.02  | Meta-LLaMA; Fudan-MOSS                         | None
2023.03  | Stanford-Alpaca, Vicuna; Zhipu-ChatGLM|OpenAI-GPT4; Baidu-ERNIE Bot; Anthropic-Claude; Google-Bard
2023.04  | Alibaba-Tongyi Qianwen; Stability AI-StableLM|SenseTime-SenseNova
2023.05  | Microsoft-Pi; TII-Falcon|iFlytek-Spark LLM; Google-PaLM2
2023.06  | Zhipu-ChatGLM2; Shanghai AI Lab-InternLM; Baichuan-BaiChuan; Hubo-TigerBot|360-Zhinao LLM
2023.07  | Meta-LLaMA2|Anthropic-Claude2; Huawei-Pangu LLM 3
2023.08  | None|ByteDance-Doubao
2023.09  | Baichuan-BaiChuan2|Google-Gemini; Tencent-Hunyuan LLM
2023.11  | 01.AI-Yi; High-Flyer-DeepSeek|xAI-Grok

At present, companies and research institutes at home and abroad are continuously launching more powerful LLMs, exploring the road toward AGI.

### 4.1.2 The Capabilities of an LLM

#### (1) Emergent Abilities

The most distinctive feature that distinguishes an LLM from a traditional PLM is that an LLM possesses `emergent abilities`. Emergent abilities refer to capabilities that are not obvious in small models but become particularly prominent in large models, despite the same model architecture and pre-training task. This can be analogized to the phase transition phenomenon in physics: the emergence of emergent abilities is like model performance rapidly improving as scale increases, surpassing random levels — in other words, what we often describe as quantitative change leading to qualitative change.

Specifically, emergent abilities can be defined as capabilities related to certain complex tasks. But generally speaking, NLP is more concerned with the general abilities they possess, that is, abilities that can be applied to solving various NLP tasks. Emergent abilities are the core reason why both industry and academia maintain such high enthusiasm and attention toward LLMs. That is, although there is still a considerable gap between the current capabilities of LLMs, the tasks they can solve, and the artificial general intelligence that humanity ultimately hopes for, thanks to emergent abilities we believe that, as research continues to deepen, high-quality data continues to emerge, and more efficient model architectures and training frameworks appear, LLMs will eventually possess the capabilities required by artificial general intelligence, thereby bringing a qualitative change to human life.

#### (2) In-context Learning

In-context learning was first introduced by GPT-3. Specifically, in-context learning allows a language model to perform a task by understanding the context and generating the corresponding output when given natural language instructions or multiple task examples, without requiring additional training or parameter updates.

For a traditional PLM, after costly pre-training, supervised fine-tuning on a specified downstream task is often still required. Although traditional PLMs are relatively small in size and have low compute requirements, even a BERT-like model (0.5B parameters) generally requires more than 10GB of GPU memory for supervised fine-tuning, which carries a certain compute cost. Meanwhile, the cost of the training data for supervised fine-tuning is even higher. Depending on the difficulty of the downstream task, the number of training samples required often ranges from 1k to tens of k, all of which need manual annotation, incurring considerable data acquisition costs. In contrast, LLMs with in-context learning ability often require no costly additional training or fine-tuning and can handle the vast majority of tasks through a few examples or by adjusting natural language instructions, greatly saving compute and data costs.

In-context learning is also triggering a transformation of the NLP research paradigm. In the traditional PLM era, the general paradigm for solving NLP downstream tasks was pre-training–fine-tuning: select a suitable pre-trained model and prepare supervised data for your downstream task to fine-tune it. But by using LLMs with in-context learning ability, the general paradigm has begun to shift toward Prompt Engineering, that is, adjusting the prompt to elicit the capabilities of the LLM. For example, for the vast majority of NLP tasks today, by adjusting the prompt or providing 1~5 natural language examples, GPT-4 can achieve results exceeding those of traditional PLM fine-tuning.

#### (3) Instruction Following

By fine-tuning on multi-task data described in natural language — the so-called `instruction tuning` — LLMs have been shown to perform well on unseen tasks that are likewise described in the instruction format. In other words, an instruction-tuned LLM can understand and follow unseen instructions and execute tasks according to the task instruction without having seen specific examples beforehand, demonstrating its powerful generalization ability.

Instruction-following ability means we no longer need to teach the model every single thing before it can do it. We only need to mix multiple kinds of instructions in the instruction-tuning stage to train its generalization ability, and the LLM can then handle the vast majority of human instructions, i.e., flexibly solve the problems users encounter. This is particularly evident with ChatGPT. The core reason ChatGPT has been able to achieve such popularity is precisely that it is no longer merely a theoretical model usable only for academic and industrial research, but can also broadly serve users across all walks of life. By giving ChatGPT instructions, it can write essays, write programs, grade exam papers, read newspapers, and so on.

Instruction-following ability enables LLMs to truly integrate with multiple industries, empowering every aspect of human life through artificial intelligence technology and thereby bringing a qualitative change to humanity. Whether it is the currently red-hot Agent and WorkFlow, or the all-around assistant and superintelligence that may appear in the not-so-distant future, what they fundamentally rely on is the instruction-following ability of LLMs.

#### (4) Step-by-Step Reasoning

Logical reasoning, especially complex reasoning tasks involving multiple reasoning steps, has always been a difficult challenge in NLP and an important reason why artificial intelligence has struggled to gain widespread recognition. After all, if a model cannot solve the basic "chickens and rabbits in the same cage" problem, or cannot recognize logical traps in language, it is hard to regard it as "intelligent" rather than "stupid".

However, traditional NLP models usually struggle to solve complex tasks involving multiple reasoning steps, such as math problems. LLMs, on the other hand, by adopting the Chain-of-Thought (CoT) reasoning strategy, can solve these tasks using prompting mechanisms that include intermediate reasoning steps, thereby arriving at the final answer. It is speculated that this ability may be acquired through training on code.

Step-by-step reasoning ability means LLMs can handle complex logical tasks, that is, they can solve the vast majority of problems in daily life that require logical judgment, thereby taking a solid step toward becoming a "reliable" intelligent assistant.

These unique abilities are important advantages that distinguish LLMs from traditional PLMs, and they also enable LLMs to excel at handling various tasks, making them powerful tools for solving complex problems and being applied across multiple domains. It is precisely because of the existence of emergent abilities, in-context learning, instruction following, and step-by-step reasoning that NLP researchers believe LLMs are an important path toward artificial general intelligence and toward helping human society achieve a qualitative leap in productivity. And in fact, numerous LLM-based applications already exist that aim to significantly boost productivity by leveraging the unique abilities of LLMs. For example, Microsoft's Copilot, built on GPT-4, draws on the LLM's powerful instruction-following and step-by-step reasoning abilities to assist programmers in writing code more efficiently, conveniently, and precisely through features such as code completion, code hints, and code writing, greatly improving programmers' productivity.

### 4.1.3 The Characteristics of an LLM

In addition to the core capabilities of LLMs discussed above, LLMs also possess some additional characteristics that are interesting or dangerous; these characteristics are also important current research directions for LLMs. Here we discuss a few of them:

#### (1) Multilingual Support

Multilingual and cross-lingual models were once an important research direction in NLP, but since LLMs need to use massive corpora for pre-training, and the training corpus is often itself multilingual, LLMs are inherently multilingual and cross-lingual — only, depending on the training corpus and instruction tuning, their abilities vary across different languages. Since high-quality English corpora still account for the majority, most models represented by GPT-4 have significantly stronger abilities in English than in Chinese. Although all of them can handle multiple languages, domestic models specially trained and optimized for Chinese (such as ERNIE Bot, Tongyi Qianwen, etc.) often show superior performance in Chinese-language settings.

#### (2) Long-Text Processing

Since how long a context the model can process determines part of the model's capability ceiling to a certain extent, LLMs often value long-text processing ability more than traditional PLMs do. Compared with traditional PLMs, where 512 tokens is the convention (e.g., the maximum context length of models like BERT and T5 is 512), LLMs have produced a stream of clever tricks for extending the maximum context length. Because they are trained on massive distributed training clusters, LLMs often already support 4k, 8k, or even 32k context lengths during training. Meanwhile, most LLMs adopt Rotary Positional Encoding (RoPE) (or AliBi, which likewise has extrapolation ability) as the positional encoding, giving them a certain ability for length extrapolation — that is, at inference time they can process text significantly longer than the training length. For example, InternLM was pre-trained on 32k-length contexts, but with RoPE it can handle contexts of up to 200k length. By continuously strengthening long-text processing ability, LLMs often gain stronger information reading and information summarization capabilities, thereby solving "impossible problems" such as asking an LLM to finish reading *Dream of the Red Chamber* and then write a corresponding college-entrance-exam-style essay.

#### (3) Extension to Multimodality

The powerful abilities of LLMs have also brought them strong cross-modal performance. As LLMs keep improving, adding extra parameters to the LLM to represent images and thereby using the LLM's powerful abilities to build models that support both text and image modalities has already become a successful approach. By introducing Adapter layers and image encoders and performing targeted supervised fine-tuning on image-text data, the model can acquire decent image-text question answering and even generation abilities. In the future, how to align the representations of text and images to build more powerful multimodal large models and extend the LLM's abilities to more modalities is an important research direction.

#### (4) The Lingering Problem of Hallucination

Hallucination refers to the phenomenon of an LLM fabricating false or incorrect information based on a prompt. For example, when we ask an LLM to generate an academic paper along with its reference list, it will often invent numerous papers and studies that look "perfectly serious" but are in fact completely nonexistent. Hallucination is an inherent defect of LLMs and a huge challenge for current LLM research and applications. Especially in fields that place a strong emphasis on precision and correctness, such as medicine and finance, the presence of hallucinations can cause very serious consequences. Many studies currently offer methods to weaken hallucinations, such as imposing restrictions in the prompt or guiding generation through RAG (Retrieval-Augmented Generation), but all of these can only reduce hallucinations to a certain degree and cannot eradicate them completely.

Beyond the points above, LLMs have many more characteristics worth studying, such as the three-stage LLM training pipeline that we will discuss in detail in the next section, and the self-reflective nature of LLMs; we will not enumerate them all here.

## 4.2 How to Train an LLM

In the previous section, we analyzed the definition of LLMs and their uniquely powerful abilities: through larger-scale parameters and massive training corpora, they acquire emergent abilities far beyond those of traditional pre-trained models, exhibiting powerful in-context learning, instruction following, and step-by-step reasoning, bringing a brand-new transformation to the NLP field. So, through what steps can we train an LLM with emergent abilities? And what is the difference between training an LLM and training a traditional pre-trained model?

<div align='center'>
    <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/4-figures/2-0.jpg" alt="alt text" width="90%">
    <p>Figure 4.1 The three stages of training an LLM</p>
</div>

Generally speaking, training a complete LLM requires going through the three stages shown in Figure 1 — Pretrain, SFT, and RLHF. In this section, we will discuss the three stages of training an LLM in detail, analyzing the process of each stage as well as its core difficulties and points of attention, helping readers understand theoretically what steps are needed to train an LLM.

### 4.2.1 Pretrain

Pretrain is the first step of training an LLM, and it is both the most core step and the one with the largest engineering workload. The pre-training of LLMs is very similar to that of traditional pre-trained models: it likewise uses massive amounts of unsupervised text to train randomly initialized model parameters. As we saw in Chapter 3, mainstream LLMs today almost all adopt a Decoder-Only, GPT-like architecture (the LLaMA architecture), and their pre-training task likewise inherits the classic pre-training task of the GPT model — Causal Language Model (CLM).

Causal language model modeling, consistent with the original language models, trains the model by giving it the preceding context and requiring it to predict the next token. The process and principles of CLM were discussed in detail in Chapter 3, so we will not repeat them here. The core difference between LLM pre-training and traditional pre-trained models lies in the scale and resource consumption of pre-training.

By definition, the core characteristic of an LLM is that it has far more parameters than traditional pre-trained models, while also being pre-trained on much more massive corpora. The traditional pre-trained model BERT has two versions, base and large. The BERT-base model consists of 12 Encoder layers with a hidden_size of 768, using 12 heads for the multi-head attention layer, and has a total of 100 million (110M) parameters; the BERT-large model consists of 24 Encoder layers with a hidden_size of 1024, has 16 heads, and has a total of 300 million (340M) parameters. Meanwhile, BERT's pre-training used a corpus of 3.3 billion (3B) tokens and was trained for 4 days on 64 TPUs. In fact, relative to traditional deep learning models, BERT with 300 million parameters and 3.3 billion training tokens was already a behemoth with outstanding capabilities and enormous resource consumption.

But as we mentioned earlier, an LLM in the usual sense generally has tens of billions or even hundreds of billions of parameters; even the smallest LLM in the broad sense generally has at least one billion (1B) parameters. For example, taking the pioneering GPT-3 as an example, it has 96 Decoder layers, a hidden_size of 12288, and 96 heads, for a total of 175 billion (175B) parameters — nearly 3 orders of magnitude larger than BERT. Even currently popular small LLMs such as Qwen-1.8B have 24 Decoder layers, a hidden_size of 2048, and 16 attention heads, with a total parameter count of 1.8 billion (1.8B).

Model|hidden_layers|hidden_size|heads|Total parameters|Pre-training data size
----| -----------|-----------|------|---------|---------
BERT-base|12|768|12|0.1B|3B
BERT-large|24|1024|16|0.3B|3B
Qwen-1.8B|24|2048|16|1.8B|2.2T
LLaMA-7B|32|4096|32|7B|1T
GPT-3|96|12288|96|175B|300B


More importantly, LLMs often require much larger-scale pre-training corpora. According to the Scaling Law proposed by OpenAI: C ~ 6ND, where C is the amount of computation, N is the number of model parameters, and D is the number of training tokens, experiments show that the number of training tokens should be 1.7 times the number of model parameters; that is, a 175B GPT-3 needs 300B tokens for pre-training. LLaMA further proposed that using 20 times as many tokens to train the model achieves optimal results, so a 175B GPT-3 could be pre-trained on 3.5T tokens of data to reach optimal performance.

Such enormous model parameters and pre-training data mean that the compute resources required to pre-train an LLM are extremely large. In fact, even pre-training a 1B large model requires at least a multi-GPU distributed cluster, which splits the model parameters, intermediate training states, and training data through a distributed framework, and completes the training through long training runs measured in days. Generally speaking, a ten-billion-scale LLM requires 1024 A100s training for over a month, while a one-billion-scale LLM generally still requires 256 A100s training for two or three days — the compute resource consumption is very high.

For precisely this reason, distributed training frameworks have become an indispensable component of LLM training. The core ideas of distributed training frameworks are data parallelism and model parallelism. Data parallelism applies when the size of the model being trained can be accommodated by a single GPU's memory, but since increasing the training batch_size increases GPU memory overhead, a large batch_size cannot be used for training; meanwhile, the amount of training data is so large that training on a single GPU would take an unacceptably long time.

<div align='center'>
    <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/4-figures/2-1.jpg" alt="alt text" width="60%">
    <p>Figure 4.2 Model parallelism and data parallelism</p>
</div>

Therefore, as shown in Figure 4.2, model instances can be run on different GPUs and on different batches of data; after each forward pass completes, the gradients of all instances are collected and a gradient update is computed, and after the model parameters are updated, they are propagated to all instances. That is, under data parallelism, the model parameters on each GPU remain identical, and the total training batch size equals the sum of the batch sizes on each card.

However, when an LLM scales up to tens of billions of parameters, a single GPU's memory often cannot hold the complete model parameters. As shown in Figure 4.3, in this case the model can be split across multiple GPUs, with each GPU holding different layers or different parts, thereby achieving model parallelism.

<div align='center'>
    <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/4-figures/2-2.jpg" alt="alt text" width="30%">
    <p>Figure 4.3 Model parallelism</p>
</div>


Building on the ideas of data parallelism and model parallelism, a variety of more efficient distributed approaches have evolved, such as tensor parallelism, 3D parallelism, ZeRO (Zero Redundancy Optimizer), and so on. Currently, mainstream distributed training frameworks include Deepspeed, Megatron-LM, ColossalAI, etc., among which Deepspeed is the most widely used.

Deepspeed's core strategies are ZeRO and CPU-offload. ZeRO is a memory-optimized data-parallel scheme whose core idea is to optimize the memory usage of each card during data parallelism, thereby enabling support for larger-scale models. ZeRO divides the memory occupied by each card during the model training stage into two categories:

- Model States, including the model parameters, the model gradients, and the state parameters of the Adam optimizer. Assuming the model has 1M parameters, in general, under mixed-precision training, this part requires 16M of storage space, of which the Adam state parameters occupy 12M of storage.
- Residual States, i.e., memory usage other than model states, including activations, various caches, and memory fragmentation.

For the above memory usage, ZeRO proposes three progressively stronger optimization strategies:

1. ZeRO-1: sharding the Adam state parameters among the model states, i.e., each card only stores $\frac{1}{N}$ of the Adam state parameters, while other parameters still keep one full copy per card.
2. ZeRO-2: additionally sharding the model gradients, so each card only stores $\frac{1}{N}$ of the model gradients and Adam state parameters; only the model parameters keep one full copy per card.
3. ZeRO-3: sharding the model parameters as well, so each card only stores $\frac{1}{N}$ of the model gradients, model parameters, and Adam state parameters.

As can be seen, as the amount of sharded parameters continues to increase, the memory each card needs to occupy keeps decreasing. Of course, more sharding also means increased communication overhead during training; generally speaking, GPU utilization per card is highest for ZeRO-1 and lowest for ZeRO-3. Which strategy to use specifically should be determined dynamically based on the compute resources available and the size of the model to be trained.

Besides the compute resource requirements, the training data itself is also a major challenge in pre-training an LLM. Training an LLM requires at least several hundred B or even over 1T tokens of pre-training corpus. According to research, the vast majority of the knowledge an LLM commands is learned during pre-training, so in order for the trained LLM to cover as broad a range of knowledge as possible, the pre-training corpus needs to combine data from multiple sources, mixed in certain proportions. Currently, the main open-source pre-training corpora include CommonCrawl, C4, Github, Wikipedia, etc. Different LLMs often build their pre-training datasets on the basis of the open-source pre-training corpora by adding some private high-quality corpora and then applying the best mixture ratio obtained from their own experiments. In fact, data mixture has always been the "core secret recipe" of pre-training an LLM; different mixture ratios can considerably affect the final performance of the trained model. For example, the table below shows LLaMA's pre-training data and its mixture ratios:

Dataset|Percentage|Dataset size (Disk size)
-----|----|---------------------
CommonCrawl|67.0%|3.3 TB
C4|15.0%|783 GB
Github|4.5%|328 GB
Wikipedia|4.5%|83 GB
Books|4.5%|85 GB
ArXiv|2.5%|92 GB
StackExchange|2.0%|78 GB

Training a Chinese LLM makes the training data even more difficult. At present, high-quality corpora are still mostly concentrated in English; for example, Wikipedia, ArXiv, etc. in the table above are all English datasets, and even in multilingual datasets such as C4, English corpora dominate. Currently open-sourced Chinese LLMs such as ChatGLM and Baichuan have not released their pre-training datasets; the only open-source Chinese pre-training datasets are currently [SkyPile](https://huggingface.co/datasets/Skywork/SkyPile-150B) (150B) open-sourced by Kunlun Tiangong, [yayi2](https://huggingface.co/datasets/wenge-research/yayi2_pretrain_data) (100B) open-sourced by Zhongke Wenge, etc. — a clear gap compared with open-source English datasets.

The processing and cleaning of pre-training data is also an important part of LLM pre-training. Numerous studies have shown that the quality of pre-training data is often more important than its volume. Pre-training data processing generally includes the following steps:

1. Document preparation. Since massive pre-training corpora are usually obtained from the internet, natural language documents generally need to be obtained from crawled websites. Document preparation mainly includes URL filtering (filtering out harmful content based on page URLs), document extraction (extracting plain text from HTML), language selection (determining the language of the extracted text), etc.
2. Corpus filtering. The core purpose of corpus filtering is to remove low-quality, meaningless, toxic, and harmful content, such as garbled text and advertisements. There are generally two methods of corpus filtering: the model-based method, i.e., training a text classifier on a high-quality corpus for filtering; and the heuristic-based method, which generally computes quality metric values of the corpus based on human-defined quality metrics for web content and filters accordingly.
3. Corpus deduplication. Experiments have shown that a large amount of repeated text significantly harms the model's generalization ability, so corpus deduplication — removing documents with very high similarity from the training corpus — is also an indispensable step. Deduplication is generally based on hash algorithms to compute document similarity within a dataset or across datasets, removing documents whose similarity exceeds a specified threshold; it can also be done via exact matching at the sequence level based on substrings.

Currently, there are already many processed high-quality pre-training corpora and frameworks dedicated to pre-training data processing. For example, there is the pre-training dataset [RedPajama-1T](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T), collected and cleaned following the LLaMA approach, and the [SlimPajama-627B](https://huggingface.co/datasets/cerebras/SlimPajama-627B/tree/main/train) dataset, filtered and deduplicated on the basis of RedPajama; experiments have proven that the high-quality 627B SlimPajama dataset can achieve better results than the 1T RedPajama dataset.

### 4.2.2 SFT

Pre-training is the fundamental source of the LLM's powerful abilities; in fact, the vast knowledge covered by an LLM basically all comes from the pre-training corpus. The LLM's performance itself is also fundamentally determined by the pre-training work. However, pre-training endows the LLM with abilities, and a second step is still needed to bring those abilities out. A pre-trained LLM is like a scholar who has read widely but never digested anything: for any odd, obscure question, it can fluently continue the text, yet it does not know what the question itself means and can only "recite by rote". The essence of this phenomenon is that the LLM's pre-training task is the classic CLM, i.e., training its ability to predict the next token; without further fine-tuning, it cannot adapt to other downstream tasks or to user instructions.

Therefore, we also need a second step to teach this well-read student how to use its knowledge, which is SFT (Supervised Fine-Tuning). Supervised fine-tuning is essentially the "fine-tuning" part of the pre-training–fine-tuning paradigm we discussed in Chapter 3. The slight difference is that for traditional pre-trained models with limited abilities, we need to fine-tune them separately for each downstream task to train the model's performance on that task. For example, to solve a text classification problem, we need to fine-tune BERT on text classification; to solve named entity recognition, we need to fine-tune on the entity recognition task.

But when facing a powerful LLM, we often no longer construct supervised data on specified downstream tasks for fine-tuning; instead, we choose to train the model's "general instruction-following ability", which is generally done through SFT in the form of `instruction tuning`.

Instruction tuning means that the training inputs are various kinds of user instructions, and the outputs the model needs to fit are the responses we hope the model will give upon receiving that instruction. For example, one of our training samples could be:

    input:告诉我今天的天气预报？(input: Tell me today's weather forecast?)
    output:根据天气预报，今天天气是晴转多云，最高温度26摄氏度，最低温度9摄氏度，昼夜温差大，请注意保暖哦 (output: According to the forecast, today's weather is sunny turning cloudy, with a high of 26°C and a low of 9°C; the day-night temperature difference is large, so please keep warm.)

That is to say, the main goal of SFT is to let the model acquire a generalized instruction-following ability from instructions of many types and styles, i.e., the ability to understand and respond to user instructions. Therefore, similar to Pretrain, the data quality and data mixture of SFT are also important factors determining the model's instruction-following ability.

First is the amount of instruction data and its coverage. In order for the LLM to acquire a generalized instruction-following ability — i.e., to perform well on instructions it was not trained on — it is necessary to collect a large number of user instructions of various categories, along with corresponding responses, to train the LLM. Generally speaking, 500~1000 training samples on a single task can already yield a decent fine-tuning effect. However, in order for the LLM to gain a generalized instruction-following ability and perform well on many kinds of task instructions, the training dataset needs to cover many types of task instructions, and also requires a relatively large amount of training data; well-performing open-source LLMs generally have SFT data volumes on the order of several B tokens.

To improve the LLM's generalization ability, the broader the coverage of the instruction dataset, the better. **However**, the mixture ratio among the many different types of instruction data is also a major challenge in LLM training. InstructGPT trained by OpenAI (the predecessor of ChatGPT) used ten kinds of instructions derived from users' use of its API:

Instruction type|Percentage
-------|-----
Text generation|45.6%
Open-domain QA|12.4%
Brainstorming|11.2%
Chat|8.4%
Text rewriting|6.6%
Text summarization|4.2%
Text classification|3.5%
Others|3.5%
Closed-domain QA|2.6%
Text extraction|1.9%

High-quality instruction datasets are difficult to obtain. Unlike the unsupervised corpora used for pre-training, the instruction datasets used for SFT are supervised corpora: besides designing a broad and reasonable set of instructions, the responses to the instructions also need to be manually annotated, and the annotations must be of high quality. In fact, a large part of ChatGPT's success comes from its high-quality manually annotated data. However, manually annotating data is extremely expensive, and few companies open-source their manually annotated instruction datasets. To reduce data costs, some researchers have proposed methods that use ChatGPT or GPT-4 to generate instruction datasets. For example, the classic open-source instruction dataset [Alpaca](https://github.com/yizhongw/self-instruct/blob/main/human_eval/user_oriented_instructions.jsonl) was built from a set of seed prompts by using ChatGPT to generate more instructions and responses to them.

The instruction dataset used for SFT generally contains the following three keys:

```json
{
    "instruction":"即输入的用户指令", (i.e., the user instruction to input)
    "input":"执行该指令可能需要的补充输入，没有则置空", (supplementary input that may be needed to execute the instruction; leave empty if none)
    "output":"即模型应该给出的回复" (i.e., the response the model should give)
}
```

For example, if our instruction is to translate the target text "今天天气真好" (The weather is really nice today) into English, then the sample can be constructed as follows:

```json
{
    "instruction":"将下列文本翻译成英文：", (Translate the following text into English:)
    "input":"今天天气真好", (The weather is really nice today)
    "output":"Today is a nice day！"
}
```

Meanwhile, in order for the model to learn a paradigm different from pre-training, a specific format is often deliberately set during SFT. For example, LLaMA's SFT format is:

    ### Instruction:\n{{content}}\n\n### Response:\n

where content is the specific user instruction. That is, for each user instruction, it will be embedded into the content part above; the user instruction here refers not only to the "instruction" in the example above, but to the concatenation of the instruction and the input, i.e., a complete instruction the model can execute. For example, for the example above, the input LLaMA receives should be:

    ### Instruction:\n将下列文本翻译成英文：今天天气真好\n\n### Response:\n
    (### Instruction:\n Translate the following text into English: The weather is really nice today\n\n### Response:\n)

And the output it needs to fit is:

    ### Instruction:\n将下列文本翻译成英文：今天天气真好\n\n### Response:\nToday is a nice day！
    (### Instruction:\n Translate the following text into English: The weather is really nice today\n\n### Response:\nToday is a nice day！)

Note that since instruction tuning is essentially still CLM training of the model — only requiring the model to understand and respond to the instruction rather than simply predict the next token — the model's prediction target is not just the output, but input + output; it is just that the input part does not participate in the loss computation, but responding to the instruction itself is still implemented in the form of predicting the next token.

However, as the abilities of LLMs continue to strengthen, the model's multi-turn dialogue ability has gradually received attention. Multi-turn dialogue means that the model can refer to the history of previous conversations when making each response. For example, an LLM without multi-turn dialogue ability might have the following conversation record:

    用户：你好，我是开源组织 Datawhale 的成员。(User: Hello, I am a member of the open-source organization Datawhale.)
    模型：您好，请问有什么可以帮助您的吗？(Model: Hello, how can I help you?)
    用户：你知道 Datawhale 是什么吗？(User: Do you know what Datawhale is?)
    模型：不好意思，我不知道 Datawhale 是什么。(Model: Sorry, I don't know what Datawhale is.)

That is, the model cannot record the historical information the user previously mentioned or that it previously answered. For an LLM with multi-turn dialogue ability, its conversation record should look like this:

    用户：你好，我是开源组织 Datawhale 的成员。(User: Hello, I am a member of the open-source organization Datawhale.)
    模型：您好，请问有什么可以帮助您的吗？(Model: Hello, how can I help you?)
    用户：你知道 Datawhale 是什么吗？(User: Do you know what Datawhale is?)
    模型：Datawhale 是一个开源组织。(Model: Datawhale is an open-source organization.)

Whether the model supports multi-turn dialogue has nothing to do with pre-training. In fact, the model's multi-turn dialogue ability comes entirely from the SFT stage. To make the model support multi-turn dialogue, we need to construct the training data in a multi-turn dialogue format during SFT, so that the model can use previous knowledge to generate answers. Suppose the multi-turn dialogue we currently need to construct is:

    <prompt_1><completion_1><prompt_2><completion_2><prompt_3><completion_3>

There are generally three ways to construct multi-turn dialogue samples:

1. Directly take the last model response as the output and all previous conversation history as the input, directly fitting the last response:
        
        input=<prompt_1><completion_1><prompt_2><completion_2><prompt_3><completion_3>
        output= MASK...<completion_3>

2. Construct N samples from N turns of dialogue:

        input_1 = <prompt_1><completion_1>
        output_1 = <completion_1>

        input_2 = <prompt_1><completion_1><prompt_2><completion_2>
        output_2 =  MASK...<completion_2>

        input_3=<prompt_1><completion_1><prompt_2><completion_2><prompt_3><completion_3>
        output_3= MASK...<completion_3>

3. Directly require the model to predict the output of every turn of dialogue:

        input=<prompt_1><completion_1><prompt_2><completion_2><prompt_3><completion_3>
        output= <completion_1> MASK...<completion_2> MASK...<completion_3>

It is obvious that the first method loses a great deal of intermediate information, the second method causes a great deal of redundant computation, and only the third method is the most reasonable construction for multi-turn dialogue. The reason we can construct multi-turn dialogue samples in the third way is that the LLM is essentially still performing the CLM task, with unidirectional attention computation, so at prediction time it fits from left to right in sequence, and the prediction of an earlier turn's output does not affect the prediction of later turns. Currently, the vast majority of LLMs use the multi-turn dialogue form for SFT.

### 4.2.3 RLHF

RLHF stands for Reinforcement Learning from Human Feedback; it is the key step of using reinforcement learning to train LLMs. Compared with SFT, which already took shape with GPT-3, RLHF is often considered the most core breakthrough of ChatGPT relative to GPT-3. In fact, in terms of function, we can divide the training process of an LLM into two stages: pre-training and alignment. The core role of pre-training is to endow the model with massive knowledge, while so-called alignment is to make the model consistent with human values, so that it outputs what humans hope it will output. In this process, SFT aligns the LLM with human instructions, giving it instruction-following ability; RLHF, on a deeper level, aligns the LLM with human values, bringing it to the core standards of safe, useful, and harmless.

As shown in Figure 4.4, ChatGPT's technical report divides alignment into three stages; the latter two stages — training the RM and PPO training — are the steps of RLHF:

<div align='center'>
    <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/4-figures/2-3.png" alt="alt text" width="100%">
    <p>Figure 4.4 The three stages of ChatGPT training</p>
</div>


The idea of RLHF is to introduce reinforcement learning techniques, using real-time human feedback to enable the LLM to give responses that are more satisfying to humans. Reinforcement learning is a machine learning method distinct from supervised learning, mainly discussing how an agent can maximize the reward it can obtain in complex, uncertain environments. Reinforcement learning mainly consists of two parts: the agent and the environment. During the reinforcement learning process, the agent keeps taking actions and obtaining feedback from the environment, adjusting its action strategy based on the feedback. Applied to the alignment of an LLM, this means that for different questions, the LLM keeps generating corresponding responses, human annotators keep providing feedback on the LLM's responses, and thereby the LLM learns which responses humans prefer and like.

RLHF is like the LLM being a student who keeps doing homework to improve its problem-solving ability. If we regard the LLM as a highly capable student, Pretrain teaches it all the foundational knowledge, SFT teaches it how to read the questions and how to solve them, and RLHF is like real practice. The LLM keeps solving exercises using the foundational knowledge learned in Pretrain and the problem-solving skills learned in SFT, and then humans, acting as teachers, grade the LLM's exercises, letting the LLM reflect on its wrong problem-solving approaches and continually reinforce the correct ones.

As shown in the figure above, RLHF consists of two steps: training the RM and PPO training.

RM, Reward Model, is the reward model. The RM is used to fit human preferences and to give feedback to the LLM. During reinforcement learning training, for each response of the LLM, the RM gives a score, which reflects the degree to which the generated response aligns with human preferences. The LLM then performs optimization training based on the RM's scores according to the principles of reinforcement learning. Therefore, the RM is essentially a text classification model that outputs a scalar reward for a given text, very similar to the hidden-layer output in text classification tasks. In concrete implementation, the RM is often just a traditional LLM architecture (or BERT architecture) plus a classification layer, exactly identical to the LLM architecture used for text classification, except that the hidden-layer output is used rather than the final classification output.

However, when training the RM, we usually do not directly use texts and their corresponding scalar rewards to train it. Because it must align with human preferences, the preference data used for RM training is usually manually annotated. But since annotators often also differ in values, scalar rewards in numerical form tend to amplify these differences, causing inconsistent rewards for responses of the same degree during training, and the model finds it hard to fit the correct scalar rewards. Therefore, we usually rank the different responses under the same completion, and then convert the rankings into rewards.

For example, our training data is often in the form:

```json
{
    "prompt":"如果你打算从商店偷东西，你觉得早上好还是晚上好？", (If you plan to steal from a store, do you think morning or evening is better?)
    "chosen":"这是违法的事情，我不能提供建议", (That is illegal, and I cannot offer advice.)
    "rejected":"考虑晚上的人口贩运和监控摄像头的差别是件好事。..." (It's a good thing to consider the difference between human trafficking at night and surveillance cameras. Nighttime may be more conducive to avoiding surveillance cameras, but store employees at night can see you better. On the other hand, since daytime is usually less busy, it is easier to avoid being noticed by other people. In any case, you must pay close attention to the behavior of store employees and all customers. They may be able to see you even if they are not directly facing you. To be safe, it's important to closely watch everything in the store rather than being careless.)
}
```

where prompt is the user's question, chosen is the response that should be aligned and conforms to human preferences, and rejected is the response that does not conform to human preferences. During training, the prompt is concatenated with chosen and rejected respectively to form chosen_example and rejected_example, which are then fed into the model separately, outputting a scalar reward through forward propagation. The model then computes the loss by maximizing the scalar difference between chosen_example and rejected_example, and performs backpropagation to complete training.

It is worth noting that the model used for RM training is often of a different size than the final LLM. For example, OpenAI used a 175B LLM and a 6B RM. Meanwhile, the model used for the RM can be an LM after SFT, or an RM trained from scratch on preference data. Which one is better remains unsettled to this day.

After the RM training is completed, the PPO algorithm can be used for reinforcement learning training. PPO, Proximal Policy Optimization, is a classic RL algorithm. In fact, other reinforcement learning algorithms can also be used for RL training, but currently the PPO algorithm, being mature and relatively low-cost, is still the algorithm best suited to RLHF.

In the concrete PPO training process, there are four models. As shown in Figure 4.5, two LLMs and two RMs. The two LLMs are the actor model, which is fine-tuned and whose parameters are updated, and the ref model, whose parameters are not updated; both are initialized from the LLM after SFT. The two RMs are the critic model, whose parameters are updated, and the reward model, whose parameters are not updated; both are initialized from the RM trained in the previous step.

<div align='center'>
    <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/4-figures/2-4.jpg" alt="alt text" width="100%">
    <p>Figure 4.5 The PPO training process</p>
</div>

As shown in the figure above, the reinforcement learning training process using the PPO algorithm is as follows:

1. Initialize two models from the LLM after SFT as the Actor Model and Ref Model respectively; initialize two models from the trained RM as the Reward Model and Critic Model respectively;
2. Input a prompt; the Actor Model and Ref Model each generate a response to the prompt;
3. Compute the KL divergence between the Actor Response and the Ref Response: $r_{KL} = -\theta_{KL}D_{KL}(\pi_{PPO}(y|x)||\pi_{base}(y|x))$ where $\pi_{PPO}(y|x)$ is the output of the Actor Model, $\pi_{base}(y|x)$ is the output of the Ref Model, and $\theta_{KL}D_{KL}$ is the method for computing the KL divergence;
4. The Actor Response is input into the Reward Model and the Critic Model respectively for scoring, where the Reward Model outputs the scalar reward corresponding to the response, and the Critic Model also outputs the accumulated reward (i.e., the cumulative reward from position i to the end);
5. The computed KL divergence and the scores of the two models are all input into the reward function to compute the reward: $loss = -(kl_{ctl} \cdot r_{KL} + \gamma \cdot V_{t+1} - V_{t}) \log P(A_t|V_t)$, where $kl_{ctl}$ is the weight parameter controlling the influence of the KL divergence on the result, $\gamma$ is the weight parameter controlling the influence of the next time step's (i.e., the sample's) score on the result, $V_t$ is the score output by the Critic Model, and $A_t$ is the score output by the Reward Model;
6. Update the parameters of the Actor Model and the Critic Model according to the actor loss and critic loss computed by the reward function respectively; note that the parameter update methods of the Actor Model and the Critic Model are different, and we will not go into detail here; interested readers can delve further into the relevant theory of reinforcement learning.

In the above process, because four models are used, the memory usage is several times that of SFT. For example, if both our RM and LLM are 7B in size, the PPO process requires roughly 240G of memory (4 x 80G A100s, with each card using 60G) just for model loading. So why do we need as many as four models? The Actor Model and Critic Model are fairly easy to understand, but the reason we also need the Ref Model and Reward Model, whose original parameters are not updated, is to constrain the model's updates so that they do not deviate too far from the original model and lose the abilities endowed by Pretrain and SFT.

Of course, such large resource consumption and the complex training process make RLHF a stage with a very high entry barrier. Some researchers, starting from the supervised learning perspective, have proposed DPO (Direct Preference Optimization), a low-barrier drop-in replacement for RLHF. The core idea of DPO is to transform the reinforcement learning problem of RLHF into supervised learning to directly learn human preferences. By using the mapping between the reward function and the optimal policy, DPO shows that the constrained reward-maximization problem can be optimized entirely through single-stage policy training; that is, by learning the optimization objective proposed by DPO, one can directly learn human preferences without training an RM or performing reinforcement learning. Since it trains directly with supervised learning, DPO only needs two LLMs to complete training, and its training process is much simpler than PPO, making it a simpler and easier-to-use replacement for RLHF. As for why the optimization objective proposed by DPO can directly learn human preferences, the authors proved it through a series of mathematical derivations; interested readers can read further on their own, and we will not repeat it here.

Next, we will implement, in turn, how to train an LLM from scratch, including pre-training, SFT, and RLHF.

**References**

[1] Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, Ryan Lowe. (2022). *Training language models to follow instructions with human feedback.* arXiv preprint arXiv:2203.02155.

[2] Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* arXiv preprint arXiv:1810.04805.

[3] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, Dario Amodei. (2020). *Scaling Laws for Neural Language Models.* arXiv preprint arXiv:2001.08361.

[4] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Jack W. Rae, Oriol Vinyals, Laurent Sifre. (2022). *Training Compute-Optimal Large Language Models.* arXiv preprint arXiv:2203.15556.

[5] Qi Wang, Yiyuan Yang, Ji Jiang. (2022). Easy RL: Reinforcement Learning Tutorial . Beijing: Posts & Telecom Press. ISBN: 9787115584700. https://github.com/datawhalechina/easy-rl

[6] Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn. (2024). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model.* arXiv preprint arXiv:2305.18290.

[7] Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang, Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, Peiyu Liu, Jian-Yun Nie, Ji-Rong Wen. (2025). *A Survey of Large Language Models.* arXiv preprint arXiv:2303.18223.
