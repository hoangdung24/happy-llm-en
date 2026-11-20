---
title: "Chapter 3: Pre-trained Language Models"
description: "Pre-trained language models: encoder-only models like BERT, decoder-only models like GPT, and how pretraining and fine-tuning work."
---

## 3.1 Encoder-only PLM

In the previous chapter, we explained the attention mechanism that brought huge changes to the NLP field and the Transformer model built with this mechanism. The milestone shift of NLP models started there. From that explanation, we can see that the Transformer architecture has two parts. They are the Encoder and the Decoder. Each part has a different structure and different inputs and outputs.

Based on the characteristics of the Encoder and the Decoder, and on the pre-training idea of ELMo, different approaches began to emerge. These approaches optimize the Transformer. For example, Google kept only the Encoder layers, stacked them, and proposed a different pre-training task called Masked Language Model (MLM). This created the representative model that unified Natural Language Understanding (NLU) tasks: BERT. OpenAI, on the other hand, chose the Decoder layers and used the original Language Model (LM) task. By continuously increasing the model parameters and the pre-training corpus, it created the GPT series of models. That series has a clear advantage on NLG (Natural Language Generation) tasks. These models are also the base models of today's popular LLMs. Of course, there is another approach that keeps both the Encoder and the Decoder. This approach builds a pre-trained Transformer model, such as the T5 model released by Google.

In this chapter, we will introduce the major pre-trained models of the Transformer era in the order of Encoder-Only, Encoder-Decoder, and Decoder-Only. We will cover the three core model architectures, the pre-training tasks chosen by each mainstream model, and their unique advantages. This is also the model foundation of all current mainstream LLMs.

### 3.1.1 BERT

BERT, whose full name is Bidirectional Encoder Representations from Transformers, is a pre-trained language model released by the Google team in 2018. The model was published in the paper "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding". It achieved State Of The Art (SOTA) performance on seven natural language processing benchmark tasks including GLUE and MultiNLI. This was a milestone achievement. Since BERT's release, the pre-training + fine-tuning paradigm has become the mainstream approach for NLP tasks. BERT itself has continued to be updated and improved. Models built on BERT for optimization, such as MacBERT and BART, have also appeared. BERT is a landmark achievement of NLP. It marked major progress on various NLP tasks and the establishment of the dominance of pre-trained models. It was not until the birth of LLMs that the leading position in the NLP field shifted away from the BERT family of models. Even in the LLM era, BERT is an essential link for anyone who wants to deeply understand LLMs and NLP.

#### (1) Intellectual Heritage

BERT is a pre-trained model that unifies multiple ideas. The core ideas it inherits are:

- The Transformer architecture. As we introduced in the previous chapter, the 2017 paper "Attention is All You Need" proposed the Transformer model. It uses only the attention mechanism and discards RNN and LSTM structures, and it brought a new model architecture. BERT inherits the ideas of the Transformer and optimizes on the Transformer's model base. By stacking the Encoder structure to enlarge model parameters, it created a model architecture naturally gifted at NLU tasks.
- The pre-training + fine-tuning paradigm. Also in 2018, the birth of ELMo marked the birth of the pre-training + fine-tuning paradigm. The ELMo model is based on a bidirectional LSTM architecture. It is pre-trained on training data with a language model objective and then fine-tuned on downstream tasks. It showed superior performance and steered the NLP field toward the pre-training + fine-tuning research approach. BERT also adopted this paradigm. It adjusted the model architecture to the Transformer and introduced the pre-training task MLM. This task is better suited to text understanding and can capture deep bidirectional semantic relationships. With these changes, BERT pushed the pre-training + fine-tuning paradigm to its climax.

Next, we will take a deep dive into BERT from three aspects: model architecture, pre-training tasks, and downstream task fine-tuning. We will analyze BERT's core ideas and advantages. This helps everyone understand why BERT can achieve performance far beyond previous models. It also helps everyone understand more deeply how LLMs could defeat BERT and open the curtain of a new era.

#### (2) Model Architecture: Encoder Only

BERT's model architecture is formed by taking the Encoder part of the Transformer and stacking it. Its main structure is shown in Figure 3.1:

<div align="center">
  <img src="/images/3-figures/1-0.webp" alt="Figure 3.1 BERT model structure" width="1244" height="836" loading="lazy" decoding="async" />
  <p>Figure 3.1 BERT model structure</p>
</div>

BERT is a pre-trained model built for NLU tasks. Its input is generally a text sequence, and its output is generally a Label, such as the positive and negative labels of sentiment classification. The Transformer is a Seq2Seq model. BERT, built by stacking Encoders, is essentially also a Seq2Seq model. It simply does not include a Decoder tailored to specific tasks. Therefore, to adapt to various NLU tasks, a classification head, prediction_heads, is added at the very top of the model. It converts the multi-dimensional hidden states into the classification dimension through a linear layer. For example, if there are two classes in total, the output of prediction_heads is a two-dimensional vector.

The model as a whole consists of Embedding, Encoder, and prediction_heads:

<div align="center">
  <img src="/images/3-figures/1-1.webp" alt="Figure 3.2 Simplified structure of the BERT model" width="607" height="616" loading="lazy" decoding="async" />
  <p>Figure 3.2 Simplified structure of the BERT model</p>
</div>

The input text sequence first goes through the tokenizer. Each model's tokenizer operates in a similar way. You can refer to the Transformer's tokenizer mechanism, which will not be repeated later. Then the sequence enters the Embedding layer and is converted into hidden_states of a specific dimension. Then it passes through the Encoder blocks. The Encoder blocks are N stacked Encoder Layers. BERT comes in two sizes. The base version has 12 Encoder Layers, a hidden layer dimension of 768, and 110M total parameters. The large version has 24 Encoder Layers, a hidden layer dimension of 1024, and 340M total parameters. The top-level hidden_states produced after encoding by the Encoder finally passes through prediction_heads to obtain the final class probabilities. A Softmax computation yields the class predicted by the model.


> BERT uses WordPiece as its tokenization method. WordPiece is a statistics-based subword segmentation algorithm. Its core idea is to split words into subwords (e.g., "playing" -> ["play", "##ing"]). Its merge operation is based on maximizing the likelihood of a language model. For non-space-delimited languages such as Chinese, a single Chinese character is usually treated as the atomic token unit.

prediction_heads is essentially a linear layer plus an activation function. Generally, the output dimension of the last linear layer equals the number of classes of the task, as shown in Figure 3.3:

<div align="center">
  <img src="/images/3-figures/1-5.webp" alt="Figure 3.3 Structure of prediction_heads" width="127" height="245" loading="lazy" decoding="async" />
  <p>Figure 3.3 Structure of prediction_heads</p>
</div>

Each Encoder Layer is a layer with a structure similar to the Encoder Layer in the Transformer, as shown in Figure 3.4:

<div align="center">
  <img src="/images/3-figures/1-2.webp" alt="Figure 3.4 Encoder Layer structure" width="346" height="590" loading="lazy" decoding="async" />
  <p>Figure 3.4 Encoder Layer structure</p>
</div>

As shown in Figure 3.5, the hidden_states that have been mapped by the Embedding layer enter the core attention mechanism. Then they are added to the original input via a residual connection. Then they pass through an Intermediate layer to obtain the final output. The Intermediate layer is BERT's special name. It is actually just a linear layer plus an activation function:

<div align="center">
  <img src="/images/3-figures/1-3.webp" alt="Figure 3.5 Intermediate structure" width="334" height="305" loading="lazy" decoding="async" />
  <p>Figure 3.5 Intermediate structure</p>
</div>

Note that the activation function used by BERT is the GELU function. Its full name is the Gaussian Error Linear Unit activation function. This is an activation function that only began to receive wide attention with BERT. GELU is computed as:

$$GELU(x) = 0.5x(1 + tanh(\sqrt{\frac{2}{\pi}})(x + 0.044715x^3))$$

The core idea of GELU is to introduce the idea of stochastic regularization into the activation function. The function decides whether to drop or keep a neuron through the input's own probability distribution. We will not go into GELU's principles and core ideas here. Interested readers can study them on their own.

BERT's attention mechanism is almost identical to the self-attention mechanism of the Transformer's Encoder. BERT does integrate relative position encoding into the attention mechanism. It treats the relative position encoding as trainable weight parameters as well, as shown in Figure 3.6:

<div align="center">
  <img src="/images/3-figures/1-4.webp" alt="Figure 3.6 Structure of BERT&#x27;s attention mechanism" width="383" height="620" loading="lazy" decoding="async" />
  <p>Figure 3.6 Structure of BERT's attention mechanism</p>
</div>

As shown in the figure, BERT's attention computation and that of the Transformer differ in one point. After computing the attention scores, relative position information is first incorporated through a Position Embedding layer. This Position Embedding layer is actually just a linear matrix. Fitting relative positions with trainable parameters can, relatively speaking, fit richer relative position information than the absolute positional encoding (Sinusoidal) used by the Transformer. However, this also adds a fair number of model parameters. It is completely unable to handle inputs longer than the model's training length. For example, the maximum context length BERT can handle is 512 tokens.

Note: The original BERT (as proposed in the paper) uses the same absolute positional encoding as the Transformer. Later improvements, including various BERT variants, use the relative position encoding described above. To help readers understand a more complete picture of the model's structural design, the improved version of BERT is chosen here.

It can be seen that BERT's model architecture is built on top of the Transformer's Encoder. This is why we say BERT inherits the ideas of the Transformer.

#### (3) Pre-training Tasks: MLM + NSP

Compared with the model architecture, BERT's greater innovations lie in the two new pre-training tasks it proposed: MLM and NSP (Next Sentence Prediction). The model architecture basically inherits from the Transformer. The core advantage of the pre-training + fine-tuning paradigm is that it separates pre-training from fine-tuning. A model that has completed one pre-training can be applied to almost all downstream tasks through fine-tuning alone. As long as the fine-tuning cost is low, the model still has greater application value. This holds even if the pre-training cost is several or even tens of times higher than before. Therefore, the model parameters and the amount of pre-training data can be further expanded. A massive pre-training corpus lets the model fit latent semantics and underlying knowledge. In this way, the model gains strong language understanding and generation capabilities through long-time, large-scale pre-training.

Therefore, the core requirement for pre-training data is an extremely large data scale (hundreds of millions of tokens). Fully supervised data produced by manual annotation can hardly reach this scale. Therefore, pre-training data must come from unsupervised corpora. This is also why traditional pre-training tasks are all LMs. The LM approach of predicting the next text from the preceding text can be applied directly to any text. For any text, we only need to mask the following text, feed the preceding text into the model, and ask it to predict. This achieves LM training, so all text corpora on the Internet can be used for pre-training.

However, a major flaw of the LM pre-training task is that it directly fits left-to-right semantic relationships but ignores bidirectional semantic relationships. The Transformer represents position information in the text sequence through positional encoding. This is still fundamentally different from directly fitting bidirectional semantic relationships. For example, BiLSTM (bidirectional LSTM model) often outperforms LSTM models in semantic representation. This happens precisely because BiLSTM fits bidirectional semantic relationships through a bidirectional LSTM. So, is there a pre-training task that can both exploit massive unsupervised corpora and train the model's ability to fit bidirectional semantic relationships?

Based on this idea, Jacob and other scholars proposed MLM, the Masked Language Model, as a new pre-training task. The LM simulates human writing. MLM instead simulates a "cloze test". The idea of MLM is simple. In a text sequence, some tokens are randomly masked. Then all unmasked tokens go into the model, and the model must predict the masked tokens based on the input. For example, the input and output could be:

    Input: I <MASK> you because you are <MASK>
    Output: <MASK> - love; <MASK> - wonderful

The model can use both the context before and after the masked tokens to understand the semantics and predict the masked tokens. Through such a task the model can fit bidirectional semantics and thus better achieve text understanding. Likewise, the MLM task requires no artificial annotation of the text at all, only random masking. So all text corpora on the Internet can also be used for pre-training. For example, BERT's pre-training used a corpus of fully 3300M words.

However, MLM also has its inherent flaws. The LM task simulates the natural process of human creation. Its training is completely consistent with downstream tasks. During training the next text is predicted from the preceding text, and the same holds during downstream fine-tuning and inference. MLM is different. During downstream fine-tuning and inference, there are in fact no `<MASK>` tokens added by us. We directly obtain the corresponding hidden states from the original text. Then, depending on the downstream task, we feed them into a classifier or other components. The inconsistency between pre-training and fine-tuning will greatly affect the model's performance in downstream fine-tuning. To address this problem, the authors improved the MLM strategy.

During actual MLM training, 15% of the tokens in the training corpus are randomly selected for masking. These 15% of tokens are not all masked as `<MASK>`. There is an 80% probability of being masked, a 10% probability of being replaced by an arbitrary token, and a 10% probability of remaining unchanged. The 10% unchanged case exists precisely to eliminate the inconsistency between pre-training and fine-tuning. The 10% random replacement mainly serves to force the model to keep learning contextual information. If everything were masked, the model would only need to process the masked positions. It would only learn the tokens to be predicted and lose its learning of the context. By introducing some random tokens, the model cannot determine which tokens need to be predicted. It is thus forced to maintain the contextual representation distribution of every token. It thereby acquires the ability to represent the features of a sentence. Since the probability of random tokens is low, they do not affect the model's substantive language understanding ability.

Besides MLM, BERT also proposed another pre-training task: NSP, Next Sentence Prediction. The core idea of NSP targets sentence-level NLU tasks, such as question-answer matching and natural language inference. Question-answer matching means that given a question and several candidate answers, the model must find the true answer to the question. Natural language inference means that given a premise and an inference, the model judges whether the inference conforms to the premise. Such tasks all require the model to fit relations at the sentence level and judge the relationship between two sentences. They do not need the token-level semantic relations fitted by MLM. Therefore, BERT proposed the NSP task to train the model to fit sentence-level semantic relationships.

The core idea of the NSP task is to require the model to judge whether the two sentences of a sentence pair are continuous context. For example, the input and output could be:

    Input:
        Sentence A: I love you.
        Sentence B: Because you are wonderful.
    Output:
        1 (continuous context)

    Input:
        Sentence A: I love you.
        Sentence B: Because today's dinner is so nice.
    Output:
        0 (not continuous context)

By requiring the model to judge the relationship of sentence pairs, the model is forced to fit the relationships between sentences. It is thereby forced to adapt to sentence-level NLU tasks. Likewise, the positive samples of NSP can be obtained by randomly drawing any consecutive sentences from unsupervised corpora. The negative samples can be drawn randomly after shuffling sentences. One only needs to ensure that the sentences drawn are not originally consecutive. Thus there can also be a virtually unlimited amount of training data.

During actual pre-training, BERT used 800M of BooksCorpus corpus and 2500M of English Wikipedia corpus. 90% of the data was trained with a context length of 128. The remaining 10% of the data was pre-trained with a context length of 512, for a total of about 3.3B tokens of training. Its training hyperparameters are also worth noting. BERT's training corpus is 13GB in total. It was trained for 1M steps (40 epochs) with a batch size of 256. By comparison, LLMs generally train for only one epoch, using a batch size far larger than 256.

It can be seen that the amount of training data has grown exponentially compared with traditional non-pre-trained models. More massive training data requires greater computing power. The Base and Large versions of BERT respectively used 16 TPUs and 64 TPUs training for 4 days to complete.

#### (4) Downstream Task Fine-tuning

As a milestone achievement in the NLP field, one great significance of BERT is that it formally established the two-stage idea of pre-training + fine-tuning. First pre-train on massive unsupervised corpora to acquire general text understanding and generation capabilities. Then fine-tune on the corresponding downstream tasks. A key point of this idea is whether the capabilities obtained from pre-training can be quickly transferred to the corresponding downstream tasks through low-cost fine-tuning.

To this end, BERT designed more general input and output layers to adapt to transfer learning under multiple tasks. For each input text sequence, BERT adds a special token `<CLS>` at the beginning. In subsequent encoding, this token represents the state of the whole sentence, i.e., the sentence-level semantic representation. During NSP pre-training, the feature vector corresponding to this token was used as the input to the final classifier.

After pre-training is completed, each downstream task needs only a certain amount of fully supervised, manually annotated data. This data fine-tunes the pre-trained BERT on that task. Fine-tuning is essentially consistent with the strategy of updating model parameters during training. The difference is that training is done on a specific task. It uses less training data and a smaller batch_size, and the magnitude of parameter updates is smaller. For the vast majority of downstream tasks, BERT's output can be used directly. For text classification tasks, one can simply modify the final classification head in prediction_heads in the model structure. For tasks such as sequence labeling, one can integrate hidden layer vectors from multiple layers of BERT and then output the final labeling result. For text generation tasks, one can likewise take the Encoder's output and decode it directly to obtain the final generation result. Therefore, BERT can be applied very efficiently to a variety of NLP tasks.

As soon as BERT was proposed, it achieved SOTA results on 11 NLP tracks and became the undisputed hegemon of NLU. Many subsequent models that achieved better results on NLU tasks were obtained by improving on BERT. Even into the LLM era, BERT can still achieve the best results on many NLU tasks with abundant labeled data. For certain specific tasks with abundant training data and a strong emphasis on high throughput, BERT is more usable than LLMs.

### 3.1.2 RoBERTa

As an epoch-making masterpiece of NLP, BERT achieved SOTA results on multiple leaderboards at the same time. It also drove the whole NLP field to shift toward pre-trained models. On the basis of BERT, a large number of excellent Encoder-Only pre-trained models emerged from optimization in multiple directions. Most of them have model structures similar to or identical with BERT. They add optimizations in training data, pre-training tasks, training parameters, etc., to obtain pre-trained models with stronger capabilities and more impressive performance on downstream tasks. One of them is RoBERTa, also released by Facebook.

As we said earlier, a core advantage of pre-training + fine-tuning is the use of massive unsupervised corpora for pre-training. These corpora are far larger than previous training data. In the traditional deep learning paradigm, for each task we need to train a model from scratch. So we cannot use too many model parameters. Otherwise, extremely large-scale supervised data would be needed for the model to fit well, and the cost is too high. In the pre-training + fine-tuning paradigm, we can use as much training data as possible in the pre-training stage. Only one pre-trained model is needed. Afterwards fine-tuning on each downstream task requires only a small amount of supervised data. BERT used 13GB (3.3B tokens) of data for pre-training. This is an extremely huge data scale compared with traditional NLP.

But has 13GB of pre-training data allowed BERT to achieve a sufficient fit? If we use more pre-training corpus, can we further enhance the model's performance? Also, are the pre-training task and training hyperparameters chosen by BERT optimal? RoBERTa came into being to answer these questions.

#### (1) Optimization 1: Remove the NSP Pre-training Task

RoBERTa's model architecture is exactly the same as BERT's. It uses the model parameters of BERT-large (24 Encoder Layers, a hidden layer dimension of 1024, and 340M total parameters). Regarding the pre-training task, some scholars have questioned that the NSP task does not improve model performance because it is too simple. Adding it to pre-training does not clearly benefit downstream fine-tuning, and may even bring negative effects. RoBERTa set up four experimental groups:

    1. Segment-pair MLM + NSP: BERT's original pre-training task. The input is a pair of segments, each segment containing multiple sentences, to construct the NSP task.
    2. Sentence-pair MLM + NSP: one input builds a pair of sentences, and the batch is enlarged to be token-equivalent with the original input.
    3. Full-sentences MLM: the NSP task is removed, and one input consists of complete sentences sampled continuously from one or more documents. To make the input reach the maximum length (512), one input may include multiple documents.
    4. Doc-sentences MLM: the NSP task is removed, and one input may only be sampled from a single document. Likewise the batch is enlarged to be token-equivalent with the original input.

Experimental results showed that the latter two groups significantly outperformed the former two. The doc-sentences MLM group performed best when fine-tuned on downstream tasks. Therefore, RoBERTa removed NSP from pre-training and used only the MLM task.

At the same time, RoBERTa also made improvements to the MLM task itself. In BERT, the Mask operation is completed during the data processing stage. So in later pre-training, the `<MASK>` positions to be predicted for the same sample are always the same. BERT trained for 40 epochs in total. To make the model's training data more diverse, BERT randomly masked the data four times. The data the model trained on was completely identical every 10 epochs. RoBERTa moved the Mask operation into the training stage, i.e., a dynamic masking strategy. Thus the Mask positions differ in every epoch's training data. In experiments, dynamic masking only had a very slight advantage over static masking. Since dynamic masking is more efficient and easier to implement, subsequent MLM tasks basically all use dynamic masking.

#### (2) Optimization 2: Larger-scale Pre-training Data and Longer Training

RoBERTa used a much larger amount of unsupervised corpus for pre-training. In addition to the BookCorpus and English Wikipedia used by BERT, RoBERTa also used three other sources. These sources are CC-NEWS (the English news portion of the CommonCrawl dataset), OPENWEBTEXT (English web pages), and STORIES (a story-style subset of the CommonCrawl dataset). The total is 160GB of data, ten times that of BERT.

Meanwhile, RoBERTa believed that a larger batch size can both improve optimization speed and improve the end-of-task performance. Therefore, it trained with a batch size of 8K (compared with BERT's batch size of 256) for 31K steps. At this point the total number of training tokens equals BERT's 3.3B, and the model performed better. This demonstrated the significance of large batch sizes. On this basis, RoBERTa trained a total of 500K steps (about 66 epochs). RoBERTa no longer adopted BERT's strategy of doing most of the training at a length of 256 and finishing training at a length of 512. Instead, all training was done at a length of 512.

Larger pre-training data, longer sequence lengths, and more training epochs require more computing resources in the pre-training stage. To train a RoBERTa, Meta used 1024 V100s (32GB of memory) training for one day.

#### (3) Optimization 3: A Larger BPE Vocabulary

Unlike the WordPiece algorithm used by BERT, RoBERTa used BPE as the encoding strategy of its Tokenizer. BPE, Byte Pair Encoding, refers to using subword pairs as the unit of tokenization. For example, the sentence "Hello World" may be split into four subword pairs: "Hel", "lo", "Wor", "ld". For Chinese, whose basic unit is the character, splitting is generally done according to byte encoding. For example, in UTF-8 encoding, "我" is encoded as "E68891". In BPE it might be split into two token pairs, "E68" and "891".

Generally speaking, the larger the BPE encoding vocabulary, the better the encoding effect. The Embedding layer maps tokens from the vocabulary space to the hidden space. The shape of the Embedding is (vocab_size, hidden_size). So a larger vocabulary also increases the model's parameters.

BERT's original BPE vocabulary size is 30K. RoBERTa chose a vocabulary of 50K to optimize the model's encoding ability.

Through the above three optimizations, RoBERTa successfully refreshed the SOTA on multiple downstream tasks on the basis of the BERT architecture. For a time it became the most popular pre-trained model among BERT-family models. RoBERTa's success also demonstrated the importance of larger pre-training data and longer pre-training. This is one of the foundations for the birth of LLMs.

### 3.1.3 ALBERT

On the basis of BERT, RoBERTa further explored the effect of larger-scale pre-training. The ALBERT model, also optimized on the BERT architecture, explored on the angle of whether model parameters can be reduced while maintaining model capability. ALBERT optimized the model structure and improved the NSP pre-training task. In this way it achieved capability beyond BERT with a smaller parameter scale. Some of the improvement ideas proposed by ALBERT were not widely adopted in subsequent research. Still, its method of reducing model parameters and the new pre-training task it proposed, SOP, provided important reference for the NLP field.

#### (1) Optimization 1: Factorize the Embedding Parameters

Pre-trained models such as BERT have far more parameters than traditional neural networks. As mentioned above, BERT-large has 24 Encoder Layers, a hidden layer dimension of 1024, and 340M total parameters. Among these, the parameter matrix of the Embedding layer has dimension $V*H$. V is the vocabulary size of 30K and H is the hidden layer size of 1024. So the Embedding layer's parameters reach 30M. This setup brings an even bigger problem. Google tried to build wider models, i.e., with a larger hidden layer dimension. They found that increasing the hidden layer dimension causes a huge rise in the Embedding layer's parameters. If the hidden layer dimension is increased to 2048, the Embedding layer's parameters grow to 61M. This greatly increases the model's computational overhead.

From another angle, the vectors output by the Embedding layer are our dense vector representation of the text tokens. Judging from the successful experience of Word2Vec, such word vectors do not need a very large dimension. Word2Vec achieved very good results with only 100 dimensions. Therefore, the output of the Embedding layer perhaps does not need to be the same as the hidden layer size.

Therefore, ALBERT factorized the Embedding layer's parameter matrix. It decoupled the Embedding layer's output dimension from the hidden layer dimension. A linear matrix after the Embedding layer does the dimension transformation. ALBERT set the Embedding layer's output to 128. It therefore added a $128*1024$ linear matrix after the Embedding layer to raise the Embedding layer's output back up to the hidden layer size. In other words, the Embedding layer's parameters are reduced from $V*H$ to $V*E + E*H$. When E is much smaller than H, this method's optimization of the Embedding layer's parameters is very significant.

#### (2) Optimization 2: Cross-layer Parameter Sharing

By analyzing BERT's parameters, ALBERT found that the parameters of the various Encoder layers appear highly consistent. The 24 Encoder layers bring a huge number of model parameters. So ALBERT proposed letting the various Encoder layers share model parameters to reduce the model's parameter count.

In terms of concrete implementation, ALBERT simply initializes only one Encoder layer. During computation, 24 rounds of computation are still performed, but each round of computation passes through this single Encoder layer. So it is a model with 24 Encoder computations, but there is only one layer of Encoder parameters. This greatly reduces the model's parameter count. Under these conditions, the hidden layer dimension can be greatly expanded. This achieves a wider model with fewer parameters. ALBERT proved through experiments that BERT has 334M parameters. The ALBERT (xlarge version) with the same 24 Encoder layers but a hidden layer dimension of 2048 has only 59M parameters. It still performs even better than BERT.

The above optimization greatly reduces the model's parameter count and even improves the model's performance. It also has obvious shortcomings. ALBERT's parameter count is far smaller than BERT's, but its training efficiency is only slightly better than BERT's. In the model's setup, the layers share weights, but computation still has to go through 24 rounds of Encoder Layer computation. So the speed during training and inference is actually slower than BERT's. This is also an important reason why ALBERT ultimately failed to replace BERT.

#### (3) Optimization 3: Proposing the SOP Pre-training Task

Similar to RoBERTa, ALBERT also believed the NSP task is too simple to have a significant impact on the model's performance improvement during pre-training. RoBERTa chose to remove NSP directly. ALBERT instead chose to improve NSP by increasing its difficulty to optimize the model's pre-training.

In the traditional NSP task, the positive examples are sentence pairs composed of two consecutive sentences. The negative examples are sentence pairs drawn from any two documents. The model can judge positive and negative examples relatively easily and cannot learn deep semantics well. The SOP task proposes an improvement. The positive examples are likewise composed of two consecutive sentences, but the negative examples reverse the order of the two. The model must not only fit the relationship between two sentences. It must also learn their order relationship, which greatly increases the difficulty of pre-training. For example, compared with the NSP task example given earlier, the SOP task example looks like:

    Input:
        Sentence A: I love you.
        Sentence B: Because you are wonderful.
    Output:
        1 (positive sample)

    Input:
        Sentence A: Because you are wonderful.
        Sentence B: I love you.
    Output:
        0 (negative sample)

ALBERT proved through experiments that the SOP pre-training task significantly improves model performance. Models pre-trained with MLM + SOP perform better than models pre-trained with MLM alone. Models pre-trained with MLM alone in turn outperform models pre-trained with MLM + NSP.

Through the above three optimizations, ALBERT achieved stronger performance with smaller parameters. The reduced training and inference efficiency brought by its architecture limited the model's further development. Still, the idea of building wider models provided reference value for many more capable models.

As the kings of NLP in the pre-training era, BERT and BERT-family models played an extremely important role in many NLP tasks. Besides RoBERTa and ALBERT introduced above, there are many rising stars that optimized BERT from other higher angles. ERNIE further improved the pre-training task. The small model DistilBERT distills BERT. XLM focuses on multilingual tasks. This text will not go into them one by one. The Encoder-Only architecture represented by BERT is not the only variant of the Transformer. Next, we will introduce another mainstream architecture of the Transformer, more similar to the original Transformer, with T5 as the representative: the Encoder-Decoder architecture.
## 3.2 Encoder-Decoder PLM

In the previous section, we learned about models with the Encoder-Only structure. We mainly introduced BERT's model architecture, pre-training tasks, and downstream task fine-tuning. BERT is a Transformer-based Encoder-Only model that learns the bidirectional semantic relationships of text through the pre-training tasks MLM and NSP. In this way it achieves excellent performance on downstream tasks. BERT also has some problems. One is the inconsistency between the MLM task and downstream fine-tuning. Another is its inability to handle inputs longer than the model's training length. To solve these problems, researchers proposed Encoder-Decoder models. These models solve these issues by introducing the Decoder part, and at the same time they bring new ideas and methods to the NLP field.

In this section, we will learn about models with the Encoder-Decoder structure. We mainly introduce T5's model architecture and pre-training tasks, as well as the "NLP unification" idea first proposed by the T5 model.

### 3.2.1 T5

T5 (Text-To-Text Transfer Transformer) is a pre-trained language model proposed by Google. It unifies the representation of all NLP tasks as text-to-text conversion problems. This greatly simplifies model design and task handling. T5 is based on the Transformer architecture and contains two parts, the encoder and the decoder. It uses the self-attention mechanism and multi-head attention to capture global dependencies. It uses relative position encoding to handle position information in long sequences. It also includes a feed-forward neural network in each layer to further process features.

T5's unification idea represents different NLP tasks uniformly as conversion from input text to output text. These tasks include text classification, question answering, translation, etc. This approach simplifies model design, parameter sharing, and the training process. It also improves the model's generalization ability and efficiency. Through this unified processing approach, T5 reduces task-specific model tuning work. It can also use the same data processing and training framework. This greatly improves the performance of multi-task learning and the convenience of application. Next, we will introduce the T5 model from three aspects: model structure, pre-training tasks, and the unification idea.

#### (1) Model Structure: Encoder-Decoder

BERT adopted the Encoder-Only structure, containing only the encoder part. GPT adopted the Decoder-Only structure, containing only the decoder part. T5 adopts the Encoder-Decoder structure, where both the encoder and the decoder are designed based on the Transformer architecture. The encoder is used to process the input text, and the decoder is used to generate the output text. The encoder and the decoder interact through the attention mechanism. In this way they realize the conversion from input text to output text. Its main structure is shown in Figure 3.7:

<div align="center">
  <img src="/images/3-figures/2-1.webp" alt="Figure 3.7 Detailed structure of the T5 model" width="1600" height="1744" loading="lazy" decoding="async" />
  <p>Figure 3.7 Detailed structure of the T5 model</p>
</div>

As shown in Figure 3.8, viewed as a whole, T5's model structure includes the Tokenizer part and the Transformer part. The Tokenizer part is mainly responsible for converting the input text into an input format acceptable to the model. This includes operations such as tokenization and encoding. The Transformer part is further divided into EncoderLayers and DecoderLayers. These are each composed of small Blocks. Each Block contains the multi-head attention mechanism, a feed-forward neural network, and a Norm layer. The Block design makes the model more flexible. Like Lego, the number and layers of Blocks can be adjusted according to the complexity of the task and the size of the dataset.

<div align="center">
  <img src="/images/3-figures/2-2.webp" alt="Figure 3.8 Overall structure of the T5 model" width="1600" height="1452" loading="lazy" decoding="async" />
  <p>Figure 3.8 Overall structure of the T5 model</p>
</div>

Both the Encoder and the Decoder parts of the T5 model are designed based on the Transformer architecture. They mainly include two structures: Self-Attention and the feed-forward neural network. Self-Attention is used to capture the global dependencies in the input sequence. The feed-forward neural network is used to handle the non-linear transformation of features.

Unlike the Encoder, the Decoder also contains an Encoder-Decoder Attention structure. It is used to capture the dependencies between the input and output sequences. These two Attention structures are almost completely identical. They differ only in position encoding and the Mask mechanism. As shown in Figure 3.9, the structures of the Encoder and Decoder are as follows:

<div align='center'>
    <img src="/images/3-figures/2-3.webp" alt="Figure 3.9 Encoder and Decoder" width="1568" height="2048" loading="lazy" decoding="async" />
    <p>Figure 3.9 Encoder and Decoder</p>
</div>

T5's Self-Attention mechanism is the same as BERT's Attention mechanism. Both are designed based on the Self-Attention mechanism. The Self-Attention mechanism is a global dependency modeling method. It captures the global dependencies in the input sequence by computing the similarity between Query, Key, and Value. The Encoder-Decoder Attention differs only in position encoding and the Mask mechanism. It mainly distinguishes the input and output sequences. As shown in Figure 3.10, the structure of Self-Attention is as follows:

<div align='center'>
    <img src="/images/3-figures/2-4.webp" alt="Figure 3.10 Self-Attention structure" width="1496" height="2240" loading="lazy" decoding="async" />
    </p>Figure 3.10 Self-Attention structure</p>
</div>

Unlike the original Transformer model, the LayerNorm of the T5 model adopts RMSNorm. It normalizes the activation values of each hidden layer by computing the Root Mean Square of each neuron. RMSNorm's parameter setup is simpler than Layer Normalization, with only one learnable parameter. It can better adapt to different tasks and datasets. The RMSNorm function can be expressed by the following mathematical formula:

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{n}\sum_{i=1}^{n}x_i^2 + \epsilon}} \cdot \gamma
$$

where:
- $x_i$ is the $i$-th element of the input vector
- $\gamma$ is a learnable scaling parameter
- $n$ is the dimensionality of the input vector
- $\epsilon$ is a small constant for numerical stability (to avoid division by zero)

This normalization helps stabilize the learning process. It ensures that the scale of the weights does not become too large or too small. This is particularly useful in deep learning models with many layers.

#### (2) Pre-training Task

The pre-training task of the T5 model is a key component. It enables the model to learn rich language representations. The language representation ability can be transferred to various downstream tasks in subsequent fine-tuning. The dataset used for training is a large-scale text dataset containing all kinds of text data, such as Wikipedia, news, books, etc. After careful processing of the data, the 750GB dataset C4 used for training was generated. It has been open-sourced in TensorflowData.

We can briefly summarize T5's pre-training task. It mainly includes the following parts:

- Pre-training task: The pre-training task of the T5 model is MLM, also known as the BERT-style objective. Specifically, 15% of the tokens in the input text are randomly masked. The model is then asked to predict these masked tokens. This process requires no labels and can be performed on a large amount of unlabeled text.
- Input format: During pre-training, T5 converts the input text into a "text-to-text" format. For a given text sequence, some tokens are randomly selected for masking and replaced with a special placeholder (token). Then the masked token sequence is used as the model's output target.
- Pre-training dataset: T5 used its own created large-scale dataset, the "Colossal Clean Crawled Corpus" (C4). It extracted a large amount of clean English text from Common Crawl. The C4 dataset underwent a certain amount of cleaning, removing meaningless text, duplicate text, etc.
- Multi-task pre-training: T5 also tried mixing multiple tasks together for pre-training, rather than just the MLM task alone. This helps the model learn more general language representations.
- Transition from pre-training to fine-tuning: After pre-training is completed, the T5 model is fine-tuned on downstream tasks. During fine-tuning, the model is trained on task-specific datasets, and the decoding strategy is adjusted according to the task.

Through large-scale pre-training, the T5 model can learn rich language knowledge and obtain strong language representation ability, achieving excellent performance on multiple NLP tasks. Pre-training is one of the key factors in T5's success.

#### (3) The Unification Idea

One core idea of the T5 model is the "unification idea": all NLP tasks can be unified as text-to-text tasks. This idea has profound influence in the field of natural language processing. Its design philosophy is to convert all different types of NLP tasks into a unified format. These tasks include text classification, translation, text generation, question answering, etc. In this format both input and output are plain text.

For example:
- For the text classification task, the input can be "classify: 这是一个很好的产品 (This is a very good product)", and the output is "正面 (positive)".
- For the translation task, the input can be "translate English to French: How are you?", and the output is "Comment ça va?".

T5 is pre-trained on large-scale text data and then fine-tuned on specific tasks. This process is similar to models such as BERT and GPT. T5 does unify the tasks of the pre-training and fine-tuning stages into the text-to-text form, which makes it more adaptable across various tasks.

We can understand T5's unification idea more intuitively through Figure 3.11:

<div align='center'>
    <img src="/images/3-figures/2-0.webp" alt="Figure 3.11 T5's unification idea" width="1187" height="421" loading="lazy" decoding="async" />
    <p>Figure 3.11 T5's unification idea</p>
</div>

For different NLP tasks, a task description prefix is added before each input. It explicitly specifies the type of the current task. This helps the model learn the common features across different tasks during the pre-training stage. It also makes it easy to quickly adapt to specific tasks during the fine-tuning stage. For example, the task prefix can be "summarize: " for summarization tasks, or "translate English to German: " for translation tasks.

T5's unification idea unifies all NLP tasks into the text-to-text form. It simplifies the task processing pipeline and enhances the model's generality and adaptability. This idea has promoted the development of natural language processing technology. It also provided more convenient and efficient solutions for practical applications.

## 3.3 Decoder-Only PLM

The previous two sections explain two model architectures that evolved from the Transformer. These are the Encoder-Only model represented by BERT and the Encoder-Decoder model represented by T5. One can also imagine a third architecture. This is the Decoder-Only model, which a developer builds by stacking only the Decoder.

Decoder-Only is the basic architecture of today's LLMs. Almost all current LLMs are Decoder-Only models, except for non-Transformer architectures such as RWKV and Mamba. ChatGPT triggered the LLM boom, and it is the crowning achievement of the GPT series. The GPT series represents the Decoder-Only line. The LLaMA model is the basic architecture of open-source LLMs, and it also evolved from GPT's model architecture. This section will analyze the principles, architecture, and characteristics of GPT, the representative Decoder-Only model. It will also examine the current mainstream open-source LLMs and their structures and characteristics. Together with the earlier analysis of the other Transformer series models, this section will show how today's LLMs evolved from traditional PLMs. These LLMs carry great expectations and many people consider them the necessary path to AGI.

First, let us study the representative model that opened the door to the LLM world. This model is GPT, released by OpenAI.

### 3.3.1 GPT

GPT stands for Generative Pre-Training Language Model. The OpenAI team released it in 2018 as a pre-trained language model. Academia generally recognizes BERT as the representative of the pre-trained language model era. However, GPT was the model that first explicitly proposed the pre-training–fine-tuning idea. GPT proposed the concept of generic pre-training. A model pre-trains on massive unsupervised corpora and then fine-tunes on each specific task. This approach gives huge gains on these tasks. GPT did not achieve sensational results at its release. Its performance was slightly inferior to BERT, which was released shortly after it. GPT also did not make the Decoder-Only architecture the mainstream of academic research. However, the OpenAI team firmly chose to expand the pre-training data and increase the model parameters. They kept optimizing the GPT architecture. GPT-3 was released in 2020. It established the foundation of the LLM era. ChatGPT uses GPT-3 as its base model, and it opened the door of the new era. ChatGPT became the strongest competitor of the LLM era and it is currently its biggest winner.

This section uses GPT as an example to analyze GPT and the Decoder-Only models it represents. The analysis covers three aspects: model architecture, pre-training tasks, and the development history of the GPT series. This section then leads into the current mainstream LLM architecture, LLaMA.

#### (1) Model Architecture: Decoder Only

<div align='center'>
    <img src="/images/3-figures/3-0.webp" alt="Figure 3.12 GPT model structure" width="1253" height="931" loading="lazy" decoding="async" />
    <p>Figure 3.12 GPT model structure</p>
</div>

Figure 3.12 shows GPT's overall structure, and this structure is similar to BERT's. One difference is that GPT chose the Decoder to stack the model structure, while BERT chose the Encoder. The Decoder-Only structure is also naturally suited to text generation tasks. BERT was designed to fit NLU tasks more closely. By contrast, the model designs of GPT and T5 fit NLG tasks and Seq2Seq tasks better. A natural language text input first passes through the tokenizer. The tokenizer converts it into input_ids, which correspond to vocabulary indices.

The input input_ids first pass through the Embedding layer. The Positional Embedding then encodes them by position. BERT chose a trainable fully-connected layer as the positional encoding. GPT instead followed the Transformer's classic Sinusoidal positional encoding, which is absolute positional encoding through trigonometric functions. Chapter 2 analyzes the Transformer model details, so this section does not repeat them.

The Embedding layer and the Positional Embedding layer encode the input into hidden_states. The hidden_states then enter the Decoder. The first-generation GPT model, like the original Transformer model, chose 12 decoder layers. The original Transformer's Decoder layers use a dual attention layer design. GPT's Decoder layers are actually more like the Encoder layers. The model no longer receives any encoded input from an Encoder, so the Decoder layer retains only one masked attention layer. It also moves the LayerNorm layer from after the attention layer, as in the Transformer, to before the attention layer. The hidden_states enter the Decoder layer, and LayerNorm is applied first. Then the masked attention computation follows. A residual connection and another LayerNorm then send them into the MLP, which produces the final output.

The model receives no encoded result from an Encoder, so the masked attention in the Decoder layer is also a self-attention computation. For an input hidden_states, three parameter matrices generate query, key, and value. The Transformer's Decoder instead uses the Encoder's output as key and value. The subsequent attention computation process is similar to BERT's. After the model computes the attention weights, a mask matrix masks the attention weights of future tokens. This restriction lets each token attend only to the tokens before it. This process realizes the computation of masked self-attention.

GPT's MLP layer does not choose a linear matrix for feature extraction. It instead uses two one-dimensional convolution kernels. In effect, there is not much difference between the two. The hidden_states pass through N Decoder layers and are finally mapped to the vocabulary dimension through a linear matrix. The model can then convert them into natural language tokens and generate the target sequence.

#### (2) Pre-training Task: CLM

The Decoder-Only model structure is often more suitable for text generation tasks. Decoder-Only models therefore often choose the most traditional and most direct pre-training task. This task is the Causal Language Model, hereinafter referred to as CLM.

CLM is a direct extension of the N-gram language model. The N-gram language model predicts the next token based on the previous N tokens. CLM predicts the next token based on all previous tokens of a natural language sequence. The model repeats this process continuously to generate the target text sequence. CLM is therefore a classic completion form. For example, CLM's input and output can be:

    input: 今天天气 (The weather today is)
    output: 今天天气很 (The weather today is very)

    input: 今天天气很 (The weather today is very)
    output: 今天天气很好 (The weather today is very good)

Consider a task with an input target sequence length of 256 and an expected output sequence length of 256. The model repeats 256 rounds of computation based on the first 256 tokens, then 257 tokens (input plus the first predicted token), and so on. It finally generates an output text of sequence length 512. The first 256 tokens of this output text are the input, and the last 256 tokens are the model output we expect.

BERT adopted the pre-training plus fine-tuning paradigm and achieved a major breakthrough. The MLM and NSP tasks it chose can train directly on massive unsupervised corpora. CLM is an even more direct pre-training task. It naturally matches the human habit of writing natural language text, and it directly matches downstream tasks. It is more direct than the MLM task, and a model can apply it directly to any natural language text. CLM can therefore also use massive natural language corpora for large-scale pre-training.

#### (3) Development of the GPT Series Models

Since the launch of GPT-1, OpenAI has always believed in the Decoder-Only model structure and the "scale is everything" optimization approach. OpenAI keeps expanding the pre-training dataset and the model scale. It also makes small optimizations and corrections to the model to keep exploring more capable pre-trained models. GPT-1 was suppressed by BERT. GPT-2 did not attract enough attention. GPT-3 sparked emergent abilities and brought in the era of large models. ChatGPT was epoch-making. OpenAI proved the correctness of its approach through years of effort.

The table below summarizes the changes in model structure and pre-training corpus size from GPT-1 to GPT-3:

Model | Decoder Layer | Hidden_size | Number of Attention Heads | Attention Dimension | Total Parameters | Pre-training Corpus
---- | --------------|------------|------------|----------|----------|----------
GPT-1|12|3072|12|768|0.12B|5GB
GPT-2|48|6400|25|1600|1.5B|40GB
GPT-3|96|49152|96|12288|175B|570GB

GPT-1 is the pioneering work of the GPT series and the first pre-trained model that uses Decoder-Only. GPT-1's model scale and pre-training data were both small. It inherited the traditional Transformer's model structure. It uses 12 Decoder Blocks and a hidden layer dimension of 768. Its model parameter count is only 117 million (0.12B). It was pre-trained on the 5GB BooksCorpus dataset. GPT-1's parameter scale and pre-training scale are roughly comparable to BERT-base. Its performance is somewhat inferior to BERT-base. This gap is also the reason the GPT series of models failed to become the representative of the pre-trained language model era.

GPT-2 was OpenAI's product for further exploring the multi-task learning capability of pre-trained language models on the basis of GPT-1. GPT-2's model structure is roughly comparable to GPT-1's. The model parameter scale was enlarged, however, and OpenAI changed Post-Norm to Pre-Norm. Pre-Norm performs the LayerNorm computation first, then enters the attention layer computation. The number of model layers and the scale increased, so the risk of gradient vanishing and exploding kept growing. OpenAI optimized the above structures to make the model's gradients more stable.

GPT-2's core improvement was a substantial increase in the pre-training dataset and the model scale. GPT-2's Decoder Block layers reached 48. GPT-2 released four specifications of models, and this section refers only to the largest specification, the GPT-2 model. Its hidden layer dimension reached 1600, and its total parameter count reached 1.5 billion (1.5B). It was pre-trained on its own crawled 40GB WebText dataset. It surpassed the 1st generation by an order of magnitude in both model structure and pre-training size.

GPT-2's other major breakthrough was taking zero-shot (zero-shot learning) as the main goal. This goal does not fine-tune the model at all and directly asks the model to solve tasks. In the traditional pre-training–fine-tuning paradigm, solving a problem generally requires hundreds or thousands of training samples. A developer then fine-tunes a pre-trained language model on these samples to solve the problem. Zero-shot emphasizes using no training samples at all. The developer solves the problem directly by describing the problem to the pre-trained language model. The zero-shot approach is a further step beyond the pre-training–fine-tuning paradigm, and it is a more efficient natural language paradigm. In GPT-2's era, however, the model's capability was not yet sufficient to support good zero-shot results. Zero-shot and its extension, few-shot (few-shot learning), became mainstream only in the era of large models.

GPT-3 further demonstrated OpenAI's core idea of "brute force over brains" and is the pioneering work of LLMs. On the basis of GPT-2, OpenAI further increased the model scale and the amount of pre-training data. Its total parameters reached 175B, so it is a truly deserved "large language model". There were basically no major improvements in model structure. The enormous model scale did require one change. Sparse attention replaced the traditional attention mechanism. For pre-training data, OpenAI sampled separately from large corpora such as CC, WebText, and Wikipedia. The total sampled data was 45T, which is 570GB after cleaning. According to estimates, GPT-3 required training on a distributed cluster of 1024 A100s (80GB of memory) for 1 month.

GPT-3 is called the pioneering work of LLMs for two reasons. Its enormous scale highlighted emergent abilities. It also proposed an important idea, few-shot. Few-shot is an improvement on zero-shot. Researchers found that even for GPT-3 at 175B, achieving good zero-shot performance remained quite difficult. Few-shot is a compromise on zero-shot, and it aims to provide the model with a small number of examples. Those examples teach the model to complete the task. Few-shot generally adds 3–5 examples to the prompt, which is the model's input, to help the model understand. For example, for the sentiment classification task:

    zero-shot: 请你判断'这真是一个绝佳的机会'的情感是正向还是负向，如果是正向，输出1，否则输出0 (Please judge whether the sentiment of '这真是一个绝佳的机会' (This is truly a great opportunity) is positive or negative. If positive, output 1. Otherwise output 0.)

    few-shot: 请你判断'这真是一个绝佳的机会'的情感是正向还是负向，如果是正向，输出1，否则输出0。你可以参考以下示例来判断：'你的表现非常好'为1，'太糟糕了'为0，'真是一个好主意'为1。(Please judge whether the sentiment of '这真是一个绝佳的机会' (This is truly a great opportunity) is positive or negative. If positive, output 1. Otherwise output 0. You can refer to the following examples to make the judgment. '你的表现非常好' (Your performance is very good) maps to 1. '太糟糕了' (It's terrible) maps to 0. '真是一个好主意' (What a great idea) maps to 1.)

By providing the model with a small number of examples, the model can achieve performance far better than zero-shot. Few-shot is also known as In-context Learning. This name means letting the model learn the solution to the problem from the examples in the provided context. The remarkable ability GPT-3 demonstrated on few-shot brought important progress to breakthroughs in NLP. For the vast majority of tasks, a developer can make the model solve them by manually constructing 3–5 examples. This efficiency is far higher than the traditional pre-training–fine-tuning paradigm, and it makes further real-world application of NLP possible. This is the core advantage of LLMs.

OpenAI introduced the three-stage training of pre-training, instruction fine-tuning, and reinforcement learning from human feedback on the basis of the GPT series of models. It released the epoch-making ChatGPT, which triggered the boom of large models. The releases of models such as LLaMA and ChatGLM, on the basis of GPT-3 and ChatGPT, further revealed the endless potential of LLMs. The next section dissects the current universal architecture of LLMs, LLaMA.

### 3.3.2 LLaMA

The LLaMA models are a series of large pre-trained language models developed by Meta, formerly Facebook. From LLaMA-1 to LLaMA-3, the LLaMA series of models showed the evolution of large-scale pre-trained language models and their significant potential in practical applications.

#### (1) Model Architecture: Decoder Only

Like the GPT series of models, the LLaMA models are also pre-trained language models based on the Decoder-Only architecture. The overall structure of the LLaMA models is similar to that of the GPT series. They differ only in model scale and pre-training datasets. Figure 3.13 shows a schematic diagram of the LLaMA model's architecture:

<div align='center'>
    <img src="/images/3-figures/3-1.webp" alt="Figure 3.13 LLaMA-3 model structure" width="1600" height="1142" loading="lazy" decoding="async" />
    <p>Figure 3.13 LLaMA-3 model structure</p>
</div>

Like GPT, LLaMA's processing pipeline also starts with encoding the input text through the tokenizer. The tokenizer converts it into a series of input_ids. These input_ids are the data format the model can understand and process. These input_ids then go through the transformation of the embedding layer. Each input_id is mapped to a vector in a high-dimensional space, which is a word vector. The positional embedding layer also encodes the position information of the input text. This encoding ensures the model can understand the word-order context information.

The input_ids form the hidden_states through the combination of the embedding layer and the positional embedding layer. The hidden_states contain the semantic and position information of the input text, and they are the basis for the model's subsequent processing. The hidden_states are then fed into the model's decoder layers.

In the decoder layers, the hidden_states undergo a series of processing operations. These operations come from multiple decoder blocks. Each decoder block is a core component of the model, and it deeply analyzes and transforms the hidden_states. Inside each decoder block, there is first a masked self-attention layer. In this layer, the model computes the three vectors query, key, and value separately. Linear transformation of the hidden_states produces these vectors, and they are the basis for computing the attention weights. The softmax function then computes the attention score, which reflects the strength of the association between different positions. Through the attention score, the model decides how much attention it must pay when it generates the current word. It weighs the hidden_states at different positions this way. The model then multiplies the value vector with the attention score to obtain the weighted value. This is the result of the attention.

After the masked self-attention layer is completed, the hidden_states enter the MLP layer. In this multi-layer perceptron layer, the model performs further feature extraction on the hidden_states through two fully-connected layers. The first fully-connected layer maps the hidden_states to an intermediate dimension. An activation function then performs a non-linear transformation, which increases the model's non-linear capability. The second fully-connected layer maps the features back to the original hidden_states dimension.

The hidden_states are processed by multiple decoder blocks. They then go through a linear layer for the final mapping. The output dimension of this linear layer is the same as the vocabulary dimension. The model can thus generate the probability distribution of the target sequence according to the hidden_states. Methods such as sampling or greedy decoding then generate the final output sequence. This process demonstrates the sequence generation capability of the LLaMA model.

#### (2) Development History of the LLaMA Models

LLaMA-1 series:

- Meta released LLaMA-1 in February 2023, including four versions with 7B, 13B, 30B, and 65B parameters.
- These models were pre-trained on a corpus of more than 1T tokens. The largest 65B-parameter model was trained on 2,048 A100 80G GPUs for nearly 21 days.
- Its open-source nature and excellent performance made LLaMA-1 quickly one of the most popular large models in the open-source community.

LLaMA-2 series:

- In July 2023, Meta released LLaMA-2, including four versions with 7B, 13B, 34B, and 70B parameters. All versions except the 34B model are open-source.
- LLaMA-2 expanded the pre-training corpus to 2T tokens and doubled the model's context length from 2,048 to 4,096.
- It introduced techniques such as Grouped-Query Attention (GQA).

LLaMA-3 series:

- In April 2024, Meta released LLaMA-3, including two versions with 8B and 70B parameters. It also revealed that the 400B LLaMA-3 was still in training.
- LLaMA-3 supports 8K long text and adopts a tokenizer with higher encoding efficiency, with a vocabulary size of 128K.
- It used a pre-training corpus of more than 15T tokens, more than 7 times that of LLaMA-2.

The LLaMA models are renowned for their technical innovation, multiple parameter versions, large-scale pre-training, and efficient architecture design. The models support parameter counts ranging from 7 billion to tens of billions, and they adapt to application needs of different scales. LLaMA-1 quickly gained community popularity for its open-source nature and excellent performance. LLaMA-2 and LLaMA-3 further improved model performance and application scope by introducing Grouped-Query Attention and supporting longer text inputs. LLaMA-3 adopted an efficient tokenizer with a 128K vocabulary size and a massive training data of 15T tokens. It thus achieved major progress in multilingual and multi-task processing. Meta keeps attention on model safety and community support. This attention suggests that LLaMA will continue to serve as an important driving force for the development of AI technology. It will also promote technology application and innovation worldwide.

### 3.3.3 GLM

The GLM series of models, developed by Zhipu, is one of the mainstream Chinese LLMs. It includes the ChatGLM1, 2, 3 and GLM-4 series of models and covers a variety of application scenarios such as instruction understanding and code generation. It has achieved SOTA performance on multiple Chinese evaluation sets.

ChatGLM-6B is the pioneering work of the GLM series. It was the earliest open-source Chinese LLM in China in 2023. It was also the earliest LLM to propose a unique model architecture different from GPT and LLaMA. GLM has a unique and major technical significance throughout the development history of Chinese LLMs. This section describes the development of the GLM series and introduces its unique technical ideas, which differ from the GPT and LLaMA series of models.

#### (1) Model Architecture: Slight Modifications Relative to GPT

GLM was originally a general language model base proposed by the Department of Computer Science of Tsinghua University. Its core idea is to add the MLM idea on the basis of the traditional CLM pre-training task. This addition builds a unified model that performs well on both NLG and NLU tasks.

In terms of overall model structure, GLM is roughly similar to GPT. Both are Decoder-Only structures, with only three subtle differences:

1. It uses Post Norm rather than Pre Norm. Post Norm means that when computing the residual connection, the residual computation is completed first. The LayerNorm computation follows. Models such as GPT and LLaMA all use Pre Norm, which performs the LayerNorm computation first and then the residual computation. Post Norm normalizes after the residual, so it has a stronger regularization effect on the parameters. The model's robustness is therefore better. In Pre Norm, a portion of the parameters is directly added at the end, so this portion of parameters does not need regularization. This portion precisely can prevent the model's gradient explosion or gradient vanishing. Pre Norm is therefore generally considered to work better for larger-scale models. The GLM paper proposed that Post Norm can avoid numerical errors in LLMs, although mainstream LLMs still use Pre Norm.

2. It uses a single linear layer to implement the final token prediction, rather than using an MLP. This structure is simpler and more robust. It reduces the parameter count of the final output and places a larger parameter count in the model itself.

3. The activation function was changed from ReLU to GeLUs. ReLU is the traditional activation function. Its core computational logic is to remove propagation of values smaller than 0 and keep propagation of values larger than 0. The core of GeLUs is to apply a non-linear mapping to forward propagation values close to 0. This mapping ensures a non-linear output after the activation function with a certain degree of continuity.

#### (2) Pre-training Task: GLM

GLM's core innovation mainly lies in the GLM (General Language Model) task it proposed, which is also the origin of GLM's name. GLM is a pre-training method that combines the autoencoding idea and the autoregressive idea. The autoencoding idea is actually the task-learning approach of MLM. The model randomly deletes contiguous tokens from the input text and learns the deleted tokens. The autoregressive idea is actually the traditional CLM task-learning approach, which requires the model to reconstruct the contiguous tokens in order.

GLM realizes the combination of the MLM and CLM ideas by optimizing an autoregressive blank-filling task. Its core idea is this. For an input sequence, the model performs random masking similar to MLM. The masked element is not single tokens as in MLM, but a contiguous span of tokens each time. During learning, the model must use the context around the masked span to predict the masked span. Within the masked span, the model also must complete the prediction of the masked tokens in the CLM manner. For example, the input and output can be:

    Input: I <MASK> because you <MASK>
    Output: <MASK> - love you. <MASK> - are a wonderful person

This combination of the MLM and CLM ideas fits generation-type tasks that produce token by token. It also forces the model to learn the implicit relations of the input text. The model learns these relations from both front and back directions, and it thereby fits understanding-type tasks. The GLM model produced with the GLM pre-training task demonstrated superior performance beyond BERT-family models of the same scale to a certain extent:

<div align='center'>
    <img src="/images/3-figures/3-2.webp" alt="Figure 3.14 GLM pre-training task" width="892" height="580" loading="lazy" decoding="async" />
    <p>Figure 3.14 GLM pre-training task</p>
</div>

The advantages of the GLM pre-training task were mostly displayed in the pre-trained model era. After the entry into the LLM era, CLM showed advantages far beyond MLM for pre-training at super-large scale and volume. Generative models pre-trained with CLM can also surpass MLM-trained understanding models in text understanding. They achieve this result when a developer increases the model scale and expands the pre-training scale. The ChatGLM series of models therefore only used the GLM pre-training idea in the first-generation model. Starting from ChatGLM2, it returned to the traditional CLM modeling. From the perspective of the overall development path of LLMs, the GLM pre-training task seems to be a failed attempt. However, its idea of fusing CLM and MLM through ingenious design still has considerable reference significance. So does its production of a Chinese open-source native LLM at the earliest opportunity.

#### (3) Development of the GLM Family

Zhipu built the GLM model as an early pre-trained model. This model used the native GLM architecture and the native GLM pre-training task. Zhipu then followed ChatGPT's technical approach with SFT and RLHF. In March 2023, Zhipu released ChatGLM-6B. This model was the first Chinese open-source LLM, and it became the starting point for many Chinese LLM researchers. ChatGLM-6B was pre-trained on 1T of corpus and supports a context length of 2K.

In June 2023, Zhipu open-sourced ChatGLM2-6B. Compared with the first generation, ChatGLM2 extended the context length to 32K. It also achieved a substantial breakthrough in model performance through a larger pre-training scale. In ChatGLM2, however, the model architecture basically returned to the LLaMA architecture, and it introduced the MQA attention mechanism. The pre-training task also returned to the classic CLM, and the model abandoned GLM's failed attempt.

Zhipu released ChatGLM3-6B in October 2023. Compared with the second generation, it achieved the then-SOTA performance in semantics, mathematics, reasoning, code, and knowledge. The official technical report stated that ChatGLM3 had no changes in model architecture relative to the second generation. It also stated that the main optimizations came from a more diverse training dataset, more sufficient training steps, and more optimized training strategies. Another important improvement of ChatGLM3 is that it began to support function calling and a code interpreter. Developers can directly use the open-source ChatGLM3 to develop Agents, and this support gives the model broader application value.

In January 2024, Zhipu released the GLM-4 series of models supporting a 128K context. The series includes multiple types, and Zhipu claimed in evaluation that the series reached the level of GPT-4 on English benchmarks. Zhipu did not open-source GLM-4 directly. Zhipu instead open-sourced its lightweight version, the GLM-4-9B model. This model was pre-trained on a 1T-token multilingual corpus. It has a context length of 8K, and it uses the same pipeline and data as GLM-4 for post-training. With less training computation, it surpassed Llama-3-8B and supports the functions of all tools in GLM-4.

Figure 3.15 shows the evolution of the performance of the GLM series of models on benchmark sets:

<div align='center'>
    <img src="/images/3-figures/3-3.webp" alt="Figure 3.15 Benchmark performance of the GLM model family" width="1019" height="728" loading="lazy" decoding="async" />
    <p>Figure 3.15 Benchmark performance of the GLM model family</p>
</div>

The sources for this chapter are listed below.

The sources for this chapter are listed below.

[1] Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* arXiv preprint arXiv:1810.04805.

[2] Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov. (2019). *RoBERTa: A Robustly Optimized BERT Pretraining Approach.* arXiv preprint arXiv:1907.11692.

[3] Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut. (2020). *ALBERT: A Lite BERT for Self-supervised Learning of Language Representations.* arXiv preprint arXiv:1909.11942.

[4] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu. (2023). *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer.* arXiv preprint arXiv:1910.10683.

[5] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu. (2020). *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer.* Journal of Machine Learning Research, 21(140), 1–67.

[6] Alec Radford, Karthik Narasimhan. (2018). *Improving Language Understanding by Generative Pre-Training*. Retrieved from https://api.semanticscholar.org/CorpusID:49313245

[7] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, Dario Amodei. (2020). *Language Models are Few-Shot Learners.* arXiv preprint arXiv:2005.14165.

[8] 张帆, 陈安东的文章“万字长文带你梳理Llama开源家族：从Llama-1到Llama-3”，来源：https://mp.weixin.qq.com/s/5_VnzP3JmOB0D5geV5HRFg

[9] Team GLM, Aohan Zeng, Bin Xu, Bowen Wang, Chenhui Zhang, Da Yin, Dan Zhang, Diego Rojas, Guanyu Feng, Hanlin Zhao, Hanyu Lai, Hao Yu, Hongning Wang, Jiadai Sun, Jiajie Zhang, Jiale Cheng, Jiayi Gui, Jie Tang, Jing Zhang, Jingyu Sun, Juanzi Li, Lei Zhao, Lindong Wu, Lucen Zhong, Mingdao Liu, Minlie Huang, Peng Zhang, Qinkai Zheng, Rui Lu, Shuaiqi Duan, Shudan Zhang, Shulin Cao, Shuxun Yang, Weng Lam Tam, Wenyi Zhao, Xiao Liu, Xiao Xia, Xiaohan Zhang, Xiaotao Gu, Xin Lv, Xinghan Liu, Xinyi Liu, Xinyue Yang, Xixuan Song, Xunkai Zhang, Yifan An, Yifan Xu, Yilin Niu, Yuantao Yang, Yueyan Li, Yushi Bai, Yuxiao Dong, Zehan Qi, Zhaoyu Wang, Zhen Yang, Zhengxiao Du, Zhenyu Hou, and Zihan Wang. (2024). *ChatGLM: A Family of Large Language Models from GLM-130B to GLM-4 All Tools.* arXiv preprint arXiv:2406.12793.

[10] Zhengxiao Du, Yujie Qian, Xiao Liu, Ming Ding, Jiezhong Qiu, Zhilin Yang 和 Jie Tang. (2022). *GLM: General Language Model Pretraining with Autoregressive Blank Infilling.* arXiv preprint arXiv:2103.10360.
