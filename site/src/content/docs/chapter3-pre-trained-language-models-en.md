---
title: "Chapter 3: Pre-trained Language Models"
---

## 3.1 Encoder-only PLM

In the previous chapter, we explained in detail the attention mechanism that brought huge changes to the NLP field and the Transformer model built with the attention mechanism; the milestone shift of NLP models started from there. From the explanation of the Transformer in the previous chapter, we can see that the Transformer architecture mainly consists of two parts, the Encoder and the Decoder, each with a different structure and different inputs and outputs.

Based on the characteristics of the Encoder and the Decoder, and introducing the pre-training idea of ELMo, different approaches to optimizing the Transformer began to emerge. For example, Google kept only the Encoder layers, stacked them, and proposed a different pre-training task — Masked Language Model (MLM) — creating the representative model that unified Natural Language Understanding (NLU) tasks: BERT. OpenAI, on the other hand, chose the Decoder layers and used the original Language Model (LM) task; by continuously increasing the model parameters and pre-training corpus, it created the GPT series of models with a clear advantage on NLG (Natural Language Generation) tasks, which are also the base models of today's wildly popular LLMs. Of course, there is another approach that keeps both the Encoder and the Decoder to build a pre-trained Transformer model, such as the T5 model released by Google.

In this chapter, we will introduce the major pre-trained models of the Transformer era in the order of Encoder-Only, Encoder-Decoder, and Decoder-Only, covering the three core model architectures, the pre-training tasks chosen by each mainstream model, and their unique advantages. This is also the model foundation of all current mainstream LLMs.

### 3.1.1 BERT

BERT, whose full name is Bidirectional Encoder Representations from Transformers, is a pre-trained language model released by the Google team in 2018. The model was published in the paper "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" and achieved State Of The Art (SOTA) performance on seven natural language processing benchmark tasks including GLUE and MultiNLI — a truly milestone achievement. Since BERT's release, the pre-training + fine-tuning paradigm has become the mainstream approach for NLP tasks. Not only has BERT itself continued to be updated and improved, but models built on BERT for optimization, such as MacBERT and BART, have also appeared. It can be said that BERT is a landmark achievement of NLP, marking major progress on various NLP tasks and the establishment of the dominance of pre-trained models. It was not until the birth of LLMs that the leading position in the NLP field shifted away from the BERT family of models. Even in the LLM era, BERT is an essential link for anyone who wants to deeply understand LLMs and NLP.

#### (1) Intellectual Heritage

BERT is a pre-trained model that unifies multiple ideas. The core ideas it inherits include:

- The Transformer architecture. As we introduced in the previous chapter, the 2017 paper "Attention is All You Need" proposed the Transformer model, which uses only the attention mechanism and discards RNN and LSTM structures, bringing a new model architecture. BERT inherits the ideas of the Transformer, optimizes on the Transformer's model base, and, by stacking the Encoder structure to enlarge model parameters, created a model architecture naturally gifted at NLU tasks;
- The pre-training + fine-tuning paradigm. Also in 2018, the birth of ELMo marked the birth of the pre-training + fine-tuning paradigm. The ELMo model is based on a bidirectional LSTM architecture; it is pre-trained on training data with a language model objective and then fine-tuned on downstream tasks, showing superior performance and steering the NLP field toward the pre-training + fine-tuning research approach. BERT also adopted this paradigm, and by adjusting the model architecture to the Transformer and introducing the pre-training task MLM — which is better suited to text understanding and can capture deep bidirectional semantic relationships — it pushed the pre-training–fine-tuning paradigm to its climax.

Next, we will take a deep dive into BERT from three aspects: model architecture, pre-training tasks, and downstream task fine-tuning, analyzing BERT's core ideas and advantages to help everyone understand why BERT can achieve performance far beyond previous models, and thereby more deeply understand how LLMs could defeat BERT and open the curtain of a new era.

#### (2) Model Architecture — Encoder Only

BERT's model architecture is formed by taking the Encoder part of the Transformer and stacking it. Its main structure is shown in Figure 3.1:

<div align="center">
  <img src="/images/3-figures/1-0.webp" alt="图片描述" width="1244" height="836" loading="lazy" decoding="async" />
  <p>Figure 3.1 BERT model structure</p>
</div>

BERT is a pre-trained model built for NLU tasks. Its input is generally a text sequence, and its output is generally a Label, such as the positive/negative labels of sentiment classification. However, just as the Transformer is a Seq2Seq model, BERT, built by stacking Encoders, is essentially also a Seq2Seq model — it simply does not include a Decoder tailored to specific tasks. Therefore, to adapt to various NLU tasks, a classification head, prediction_heads, is added at the very top of the model to convert the multi-dimensional hidden states into the classification dimension through a linear layer (for example, if there are two classes in total, the output of prediction_heads is a two-dimensional vector).

The model as a whole consists of Embedding, Encoder, plus prediction_heads:

<div align="center">
  <img src="/images/3-figures/1-1.webp" alt="图片描述" width="607" height="616" loading="lazy" decoding="async" />
  <p>Figure 3.2 Simplified structure of the BERT model</p>
</div>

The input text sequence first goes through the tokenizer (each model's tokenizer operates in a similar way; you can refer to the Transformer's tokenizer mechanism, which will not be repeated later), then enters the Embedding layer to be converted into hidden_states of a specific dimension, and then passes through the Encoder blocks. The Encoder blocks are N stacked Encoder Layers. BERT comes in two sizes: the base version (12 Encoder Layers, a hidden layer dimension of 768, and 110M total parameters) and the large version (24 Encoder Layers, a hidden layer dimension of 1024, and 340M total parameters). The top-level hidden_states produced after encoding by the Encoder finally passes through prediction_heads to obtain the final class probabilities, and a Softmax computation yields the class predicted by the model.


> BERT uses WordPiece as its tokenization method. WordPiece is a statistics-based subword segmentation algorithm whose core idea is to split words into subwords (e.g., "playing" -> ["play", "##ing"]). Its merge operation is based on maximizing the likelihood of a language model. For non-space-delimited languages such as Chinese, a single Chinese character is usually treated as the atomic token unit.

prediction_heads is essentially a linear layer plus an activation function. Generally, the output dimension of the last linear layer equals the number of classes of the task, as shown in Figure 3.3:

<div align="center">
  <img src="/images/3-figures/1-5.webp" alt="图片描述" width="127" height="245" loading="lazy" decoding="async" />
  <p>Figure 3.3 Structure of prediction_heads</p>
</div>

Each Encoder Layer is a layer whose structure is similar to the Encoder Layer in the Transformer, as shown in Figure 3.4:

<div align="center">
  <img src="/images/3-figures/1-2.webp" alt="图片描述" width="346" height="590" loading="lazy" decoding="async" />
  <p>Figure 3.4 Encoder Layer structure</p>
</div>

As shown in Figure 3.5, the hidden_states that have been mapped by the Embedding layer enter the core attention mechanism, then are added to the original input via a residual connection, and then pass through an Intermediate layer to obtain the final output. The Intermediate layer is BERT's special name; it is actually just a linear layer plus an activation function:

<div align="center">
  <img src="/images/3-figures/1-3.webp" alt="图片描述" width="334" height="305" loading="lazy" decoding="async" />
  <p>Figure 3.5 Intermediate structure</p>
</div>

Note that the activation function used by BERT is the GELU function, whose full name is the Gaussian Error Linear Unit activation function; this is an activation function that only began to receive wide attention with BERT. GELU is computed as:

$$GELU(x) = 0.5x(1 + tanh(\sqrt{\frac{2}{\pi}})(x + 0.044715x^3))$$

The core idea of GELU is to introduce the idea of stochastic regularization into the activation function, deciding whether to drop or keep a neuron through the input's own probability distribution. We will not go into GELU's principles and core ideas here; interested readers can study them on their own.

BERT's attention mechanism is almost identical to the self-attention mechanism of the Transformer's Encoder, but BERT integrates relative position encoding into the attention mechanism, treating the relative position encoding as trainable weight parameters as well, as shown in Figure 3.6:

<div align="center">
  <img src="/images/3-figures/1-4.webp" alt="图片描述" width="383" height="620" loading="lazy" decoding="async" />
  <p>Figure 3.6 Structure of BERT's attention mechanism</p>
</div>

As shown in the figure, the only difference between BERT's attention computation and that of the Transformer is that, after computing the attention scores, relative position information is first incorporated through a Position Embedding layer. This Position Embedding layer is actually just a linear matrix. Fitting relative positions with trainable parameters can, relatively speaking, fit richer relative position information than the absolute positional encoding (Sinusoidal) used by the Transformer. However, this also adds a fair number of model parameters, and it is completely unable to handle inputs longer than the model's training length (for example, the maximum context length BERT can handle is 512 tokens).

Note: The original BERT (as proposed in the paper) uses the same absolute positional encoding as the Transformer, while later improvements (including various BERT variants) use the relative position encoding described above. To help readers understand a more complete picture of the model's structural design, the improved version of BERT is chosen here.

It can be seen that BERT's model architecture is built on top of the Transformer's Encoder, which is why we say BERT inherits the ideas of the Transformer.

#### (3) Pre-training Tasks — MLM + NSP

Compared with the model architecture, which basically inherits from the Transformer, BERT's greater innovations lie in the two new pre-training tasks it proposed — MLM and NSP (Next Sentence Prediction). The core advantage of the pre-training–fine-tuning paradigm is that, by separating pre-training from fine-tuning, a model that has completed one pre-training can be applied to almost all downstream tasks through fine-tuning alone. As long as the fine-tuning cost is low, even if the pre-training cost is several or even tens of times higher than before, the model still has greater application value. Therefore, the model parameters and the amount of pre-training data can be further expanded, using a massive pre-training corpus to let the model fit latent semantics and underlying knowledge, so that the model gains strong language understanding and generation capabilities through long-time, large-scale pre-training.

Therefore, the core requirement for pre-training data is an extremely large data scale (hundreds of millions of tokens). Without a doubt, fully supervised data produced by manual annotation can hardly reach this scale. Therefore, pre-training data must come from unsupervised corpora. This is also why traditional pre-training tasks are all LMs — the LM's approach of predicting the next text from the preceding text can be applied directly to any text. For any text, we only need to mask the following text, feed the preceding text into the model, and ask it to predict, thereby achieving LM training; thus all text corpora on the Internet can be used for pre-training.

However, a major flaw of the LM pre-training task is that it directly fits left-to-right semantic relationships but ignores bidirectional semantic relationships. Although the Transformer represents position information in the text sequence through positional encoding, this is still fundamentally different from directly fitting bidirectional semantic relationships. For example, BiLSTM (bidirectional LSTM model) often outperforms LSTM models in semantic representation, precisely because BiLSTM fits bidirectional semantic relationships through a bidirectional LSTM. So, is there a pre-training task that can both exploit massive unsupervised corpora and train the model's ability to fit bidirectional semantic relationships?

Based on this idea, Jacob and other scholars proposed MLM, the Masked Language Model, as a new pre-training task. Compared with the LM, which simulates human writing, MLM simulates a "cloze test". The idea of MLM is simple: in a text sequence, randomly mask some tokens, then input all unmasked tokens into the model and require the model to predict the masked tokens based on the input. For example, the input and output could be:

    Input: I <MASK> you because you are <MASK>
    Output: <MASK> - love; <MASK> - wonderful

Because the model can use both the context before and after the masked tokens to understand the semantics and predict the masked tokens, through such a task the model can fit bidirectional semantics and thus better achieve text understanding. Likewise, the MLM task requires no artificial annotation of the text at all — only random masking — so all text corpora on the Internet can also be used for pre-training. For example, BERT's pre-training used a corpus of fully 3300M words.

However, MLM also has its inherent flaws. The LM task simulates the natural process of human creation; its training is completely consistent with downstream tasks, meaning that during training the next text is predicted from the preceding text, and the same holds during downstream fine-tuning and inference. But MLM is different: during downstream fine-tuning and inference, there are in fact no `<MASK>` tokens added by us; we directly obtain the corresponding hidden states from the original text and then, depending on the downstream task, feed them into a classifier or other components. The inconsistency between pre-training and fine-tuning will greatly affect the model's performance in downstream fine-tuning. To address this problem, the authors improved the MLM strategy.

During actual MLM training, 15% of the tokens in the training corpus are randomly selected for masking. But these 15% of tokens are not all masked as `<MASK>`: there is an 80% probability of being masked, a 10% probability of being replaced by an arbitrary token, and a 10% probability of remaining unchanged. The 10% unchanged case exists precisely to eliminate the inconsistency between pre-training and fine-tuning, while the 10% random replacement mainly serves to force the model to keep learning contextual information. Because if everything were masked, the model would only need to process the masked positions, thus only learning the tokens to be predicted and losing its learning of the context. By introducing some random tokens, the model cannot determine which tokens need to be predicted, and is thus forced to maintain the contextual representation distribution of every token, thereby acquiring the ability to represent the features of a sentence. Moreover, since the probability of random tokens is low, they do not affect the model's substantive language understanding ability.

Besides MLM, BERT also proposed another pre-training task — NSP, Next Sentence Prediction. The core idea of NSP targets sentence-level NLU tasks, such as question-answer matching and natural language inference. Question-answer matching means: given a question and several candidate answers, the model must find the true answer to the question; natural language inference means: given a premise and an inference, judge whether the inference conforms to the premise. Such tasks all require the model to fit relations at the sentence level and judge the relationship between two sentences, rather than the token-level semantic relations fitted by MLM. Therefore, BERT proposed the NSP task to train the model to fit sentence-level semantic relationships.

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

By requiring the model to judge the relationship of sentence pairs, the model is forced to fit the relationships between sentences, in order to adapt to sentence-level NLU tasks. Likewise, since the positive samples of NSP can be obtained by randomly drawing any consecutive sentences from unsupervised corpora, and the negative samples can be drawn randomly after shuffling sentences (one only needs to ensure that the sentences drawn are not originally consecutive), there can also be a virtually unlimited amount of training data.

During actual pre-training, BERT used 800M of BooksCorpus corpus and 2500M of English Wikipedia corpus; 90% of the data was trained with a context length of 128, and the remaining 10% of the data was pre-trained with a context length of 512, for a total of about 3.3B tokens of training. Its training hyperparameters are also worth noting: BERT's training corpus is 13GB in total, and it was trained for 1M steps (40 epochs) with a batch size of 256. By comparison, LLMs generally train for only one epoch, using a batch size far larger than 256.

It can be seen that compared with traditional non-pre-trained models, the amount of training data has grown exponentially. Of course, more massive training data requires greater computing power; the Base and Large versions of BERT respectively used 16 TPUs and 64 TPUs training for 4 days to complete.

#### (4) Downstream Task Fine-tuning

As a milestone achievement in the NLP field, one great significance of BERT is that it formally established the two-stage idea of pre-training–fine-tuning: first pre-train on massive unsupervised corpora to acquire general text understanding and generation capabilities, then fine-tune on the corresponding downstream tasks. A key point of this idea is whether the powerful capabilities obtained from pre-training can be quickly transferred to the corresponding downstream tasks through low-cost fine-tuning.

To this end, BERT designed more general input and output layers to adapt to transfer learning under multiple tasks. For each input text sequence, BERT adds a special token `<CLS>` at the beginning. In subsequent encoding, this token represents the state of the whole sentence, i.e., the sentence-level semantic representation. During NSP pre-training, the feature vector corresponding to this token was used as the input to the final classifier.

After pre-training is completed, for each downstream task, one only needs a certain amount of fully supervised, manually annotated data to fine-tune the pre-trained BERT on that task. Fine-tuning is essentially consistent with the strategy of updating model parameters during training, except that training is done on a specific task, with less training data and a smaller batch_size, and the magnitude of parameter updates is smaller. For the vast majority of downstream tasks, BERT's output can be used directly. For example, for text classification tasks, one can simply modify the final classification head in prediction_heads in the model structure. For tasks such as sequence labeling, one can integrate hidden layer vectors from multiple layers of BERT and then output the final labeling result. For text generation tasks, one can likewise take the Encoder's output and decode it directly to obtain the final generation result. Therefore, BERT can be applied very efficiently to a variety of NLP tasks.

As soon as BERT was proposed, it achieved SOTA results on 11 NLP tracks and became the undisputed hegemon of NLU; many subsequent models that achieved better results on NLU tasks were obtained by improving on BERT. Even into the LLM era, BERT can still achieve the best results on many NLU tasks with abundant labeled data. In fact, for certain specific tasks with abundant training data and a strong emphasis on high throughput, BERT is more usable than LLMs.

### 3.1.2 RoBERTa

As an epoch-making masterpiece of NLP that achieved SOTA results on multiple leaderboards at the same time, BERT also drove the whole NLP field to shift toward pre-trained models. On the basis of BERT, optimizing in multiple directions, a large number of excellent Encoder-Only pre-trained models emerged. Most of them have model structures similar to or identical with BERT, with optimizations in training data, pre-training tasks, training parameters, etc., to obtain pre-trained models with stronger capabilities and more impressive performance on downstream tasks. One of them is RoBERTa, also released by Facebook.

As we said earlier, a core advantage of pre-training–fine-tuning is that one can use massive unsupervised corpora far larger than previous training data for pre-training. In the traditional deep learning paradigm, for each task we need to train a model from scratch, so we cannot use too many model parameters; otherwise, extremely large-scale supervised data would be needed for the model to fit well, and the cost is too high. But in the pre-training–fine-tuning paradigm, we can use as much training data as possible in the pre-training stage; only one pre-trained model is needed, and afterwards fine-tuning on each downstream task requires only a small amount of supervised data. BERT used 13GB (3.3B tokens) of data for pre-training, which is an extremely huge data scale compared with traditional NLP.

But has 13GB of pre-training data allowed BERT to achieve a sufficient fit? If we use more pre-training corpus, can we further enhance the model's performance? Moreover, are the pre-training task and training hyperparameters chosen by BERT optimal? RoBERTa came into being to answer these questions.

#### (1) Optimization 1: Remove the NSP Pre-training Task

RoBERTa's model architecture is exactly the same as BERT's, namely it uses the model parameters of BERT-large (24 Encoder Layers, a hidden layer dimension of 1024, and 340M total parameters). Regarding the pre-training task, some scholars have questioned that the NSP task does not improve model performance because it is too simple; adding it to pre-training does not clearly benefit downstream fine-tuning, and may even bring negative effects. RoBERTa set up four experimental groups:

    1. Segment-pair MLM + NSP: BERT's original pre-training task, where the input is a pair of segments, each segment containing multiple sentences, to construct the NSP task;
    2. Sentence-pair MLM + NSP: one input builds a pair of sentences, and the batch is enlarged to be token-equivalent with the original input;
    3. Full-sentences MLM: the NSP task is removed, and one input consists of complete sentences sampled continuously from one or more documents; to make the input reach the maximum length (512), one input may include multiple documents;
    4. Doc-sentences MLM: the NSP task is removed, and one input may only be sampled from a single document; likewise the batch is enlarged to be token-equivalent with the original input.

Experimental results showed that the latter two groups significantly outperformed the former two, and the doc-sentences MLM group performed best when fine-tuned on downstream tasks. Therefore, RoBERTa removed NSP from pre-training and used only the MLM task.

At the same time, RoBERTa also made improvements to the MLM task itself. In BERT, the Mask operation is completed during the data processing stage, so in later pre-training, the `<MASK>` positions to be predicted for the same sample are always the same. Since BERT trained for 40 epochs in total, to make the model's training data more diverse, BERT randomly masked the data four times, i.e., the data the model trained on was completely identical every 10 epochs. RoBERTa moved the Mask operation into the training stage, i.e., a dynamic masking strategy, so that the Mask positions differ in every epoch's training data. In experiments, dynamic masking only had a very slight advantage over static masking, but because dynamic masking is more efficient and easier to implement, subsequent MLM tasks basically all use dynamic masking.

#### (2) Optimization 2: Larger-scale Pre-training Data and Longer Training

RoBERTa used a much larger amount of unsupervised corpus for pre-training. In addition to the BookCorpus and English Wikipedia used by BERT, it also used CC-NEWS (the English news portion of the CommonCrawl dataset), OPENWEBTEXT (English web pages), and STORIES (a story-style subset of the CommonCrawl dataset), for a total of 160GB of data — ten times that of BERT.

Meanwhile, RoBERTa believed that a larger batch size can both improve optimization speed and improve the end-of-task performance. Therefore, when training with a batch size of 8K (compared with BERT's batch size of 256) for 31K steps — i.e., when the total number of training tokens equals BERT's 3.3B — the model performed better, thus demonstrating the significance of large batch sizes. On this basis, RoBERTa trained a total of 500K steps (about 66 epochs). At the same time, RoBERTa no longer adopted BERT's strategy of doing most of the training at a length of 256 and finishing training at a length of 512; instead, all training was done at a length of 512.

Of course, larger pre-training data, longer sequence lengths, and more training epochs require more computing resources in the pre-training stage. To train a RoBERTa, Meta used 1024 V100s (32GB of memory) training for one day.

#### (3) Optimization 3: A Larger BPE Vocabulary

Unlike the WordPiece algorithm used by BERT, RoBERTa used BPE as the encoding strategy of its Tokenizer. BPE, Byte Pair Encoding, refers to using subword pairs as the unit of tokenization. For example, the sentence "Hello World" may be split into four subword pairs: "Hel", "lo", "Wor", "ld". For Chinese, whose basic unit is the character, splitting is generally done according to byte encoding. For example, in UTF-8 encoding, "我" is encoded as "E68891"; in BPE it might be split into two token pairs, "E68" and "891".

Generally speaking, the larger the BPE encoding vocabulary, the better the encoding effect. Of course, since the Embedding layer maps tokens from the vocabulary space to the hidden space (i.e., the shape of the Embedding is (vocab_size, hidden_size)), a larger vocabulary also increases the model's parameters.

BERT's original BPE vocabulary size is 30K; RoBERTa chose a vocabulary of 50K to optimize the model's encoding ability.

Through the above three optimizations, RoBERTa successfully refreshed the SOTA on multiple downstream tasks on the basis of the BERT architecture, and for a time became the most popular pre-trained model among BERT-family models. At the same time, RoBERTa's success also demonstrated the importance of larger pre-training data and longer pre-training, which is one of the foundations for the birth of LLMs.

### 3.1.3 ALBERT

On the basis of BERT, RoBERTa further explored the effect of larger-scale pre-training. The ALBERT model, also optimized on the BERT architecture, explored from the angle of whether model parameters could be reduced while maintaining model capability. By optimizing the model structure and improving the NSP pre-training task, ALBERT successfully achieved capability beyond BERT with a smaller parameter scale. Although some of the improvement ideas proposed by ALBERT were not widely adopted in subsequent research, its method of reducing model parameters and the new pre-training task it proposed, SOP, still provided important reference for the NLP field.

#### (1) Optimization 1: Factorize the Embedding Parameters

Pre-trained models such as BERT have far more parameters than traditional neural networks. As mentioned above, BERT-large has 24 Encoder Layers, a hidden layer dimension of 1024, and 340M total parameters. Among these, the parameter matrix of the Embedding layer has dimension $V*H$, where V is the vocabulary size of 30K and H is the hidden layer size of 1024, i.e., the Embedding layer's parameters reach 30M. Moreover, this setup brings an even bigger problem: when Google tried to build wider models (i.e., with a larger hidden layer dimension), they found that increasing the hidden layer dimension causes a huge rise in the Embedding layer's parameters; if the hidden layer dimension is increased to 2048, the Embedding layer's parameters would swell to 61M, which undoubtedly greatly increases the model's computational overhead.

From another angle, the vectors output by the Embedding layer are our dense vector representation of the text tokens. Judging from the successful experience of Word2Vec, such word vectors do not need a very large dimension; Word2Vec achieved very good results with only 100 dimensions. Therefore, the output of the Embedding layer perhaps does not need to be the same as the hidden layer size.

Therefore, ALBERT factorized the Embedding layer's parameter matrix, decoupling the Embedding layer's output dimension from the hidden layer dimension, i.e., adding a linear matrix after the Embedding layer for dimension transformation. ALBERT set the Embedding layer's output to 128, and therefore added a $128*1024$ linear matrix after the Embedding layer to raise the Embedding layer's output back up to the hidden layer size. In other words, the Embedding layer's parameters are reduced from $V*H$ to $V*E + E*H$; when E is much smaller than H, this method's optimization of the Embedding layer's parameters is very significant.

#### (2) Optimization 2: Cross-layer Parameter Sharing

By analyzing BERT's parameters, ALBERT found that the parameters of the various Encoder layers appear highly consistent. Since the 24 Encoder layers bring a huge number of model parameters, ALBERT proposed letting the various Encoder layers share model parameters to reduce the model's parameter count.

In terms of concrete implementation, ALBERT simply initializes only one Encoder layer. During computation, 24 rounds of computation are still performed, but each round of computation passes through this single Encoder layer. Therefore, although it is a model with 24 Encoder computations, there is only one layer of Encoder parameters, greatly reducing the model's parameter count. Under these conditions, the hidden layer dimension can be greatly expanded, achieving a wider model with fewer parameters. ALBERT proved through experiments that, compared with BERT's 334M parameters, the ALBERT (xlarge version) with the same 24 Encoder layers but a hidden layer dimension of 2048 has only 59M parameters, yet performs even better than BERT.

However, although the above optimization greatly reduces the model's parameter count and even improves the model's performance, it also has obvious shortcomings. Although ALBERT's parameter count is far smaller than BERT's, its training efficiency is only slightly better than BERT's, because in the model's setup, although the layers share weights, computation still has to go through 24 rounds of Encoder Layer computation, meaning that the speed during training and inference is actually slower than BERT's. This is also an important reason why ALBERT ultimately failed to replace BERT.

#### (3) Optimization 3: Proposing the SOP Pre-training Task

Similar to RoBERTa, ALBERT also believed the NSP task is too simple to have a significant impact on the model's performance improvement during pre-training. But unlike RoBERTa, which chose to remove NSP directly, ALBERT chose to improve NSP by increasing its difficulty to optimize the model's pre-training.

In the traditional NSP task, the positive examples are sentence pairs composed of two consecutive sentences, while the negative examples are sentence pairs drawn from any two documents; the model can judge positive and negative examples relatively easily and cannot learn deep semantics well. The improvement proposed by the SOP task is that the positive examples are likewise composed of two consecutive sentences, but the negative examples reverse the order of the two. That is, the model must not only fit the relationship between two sentences, but also learn their order relationship, which greatly increases the difficulty of pre-training. For example, compared with the NSP task example given earlier, the SOP task example looks like:

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

ALBERT proved through experiments that the SOP pre-training task significantly improves model performance. Models pre-trained with MLM + SOP perform better than models pre-trained with MLM alone, which in turn outperform models pre-trained with MLM + NSP.

Through the above three optimizations, ALBERT successfully achieved stronger performance with smaller parameters. Although the reduced training and inference efficiency brought by its architecture limited the model's further development, the idea of building wider models still provided reference value for many more powerful models.

As the kings of NLP in the pre-training era, BERT and BERT-family models played an extremely important role in many NLP tasks. Besides RoBERTa and ALBERT introduced above, there are many rising stars that optimized BERT from other higher angles, including ERNIE which further improved the pre-training task, the small model DistilBERT which distills BERT, XLM which focuses on multilingual tasks, etc.; this text will not go into them one by one. The Encoder-Only architecture represented by BERT is not the only variant of the Transformer. Next, we will introduce another mainstream architecture of the Transformer, more similar to the original Transformer, with T5 as the representative: the Encoder-Decoder architecture.
## 3.2 Encoder-Decoder PLM

In the previous section, we learned about models with the Encoder-Only structure, mainly introducing BERT's model architecture, pre-training tasks, and downstream task fine-tuning. BERT is a Transformer-based Encoder-Only model that learns the bidirectional semantic relationships of text through the pre-training tasks MLM and NSP, thereby achieving excellent performance on downstream tasks. However, BERT also has some problems, such as the inconsistency between the MLM task and downstream fine-tuning, and its inability to handle inputs longer than the model's training length. To solve these problems, researchers proposed Encoder-Decoder models, solving these issues by introducing the Decoder part, and at the same time bringing new ideas and methods to the NLP field.

In this section, we will learn about models with the Encoder-Decoder structure, mainly introducing T5's model architecture and pre-training tasks, as well as the "NLP unification" idea first proposed by the T5 model.

### 3.2.1 T5

T5 (Text-To-Text Transfer Transformer) is a pre-trained language model proposed by Google. By unifying the representation of all NLP tasks as text-to-text conversion problems, it greatly simplifies model design and task handling. T5 is based on the Transformer architecture and contains two parts, the encoder and the decoder; it uses the self-attention mechanism and multi-head attention to capture global dependencies, uses relative position encoding to handle position information in long sequences, and includes a feed-forward neural network in each layer to further process features.

T5's unification idea represents different NLP tasks, such as text classification, question answering, translation, etc., uniformly as conversion from input text to output text. This approach simplifies model design, parameter sharing, and the training process, and improves the model's generalization ability and efficiency. Through this unified processing approach, T5 not only reduces task-specific model tuning work, but can also use the same data processing and training framework, greatly improving the performance of multi-task learning and the convenience of application. Next, we will introduce the T5 model from three aspects: model structure, pre-training tasks, and the unification idea.

#### (1) Model Structure: Encoder-Decoder

BERT adopted the Encoder-Only structure, containing only the encoder part; GPT adopted the Decoder-Only structure, containing only the decoder part. T5 adopts the Encoder-Decoder structure, where both the encoder and the decoder are designed based on the Transformer architecture. The encoder is used to process the input text, and the decoder is used to generate the output text. The encoder and the decoder interact through the attention mechanism, thereby realizing the conversion from input text to output text. Its main structure is shown in Figure 3.7:

<div align="center">
  <img src="/images/3-figures/2-1.webp" alt="图片描述" width="1600" height="1744" loading="lazy" decoding="async" />
  <p>Figure 3.7 Detailed structure of the T5 model</p>
</div>

As shown in Figure 3.8, viewed as a whole, T5's model structure includes the Tokenizer part and the Transformer part. The Tokenizer part is mainly responsible for converting the input text into an input format acceptable to the model, including operations such as tokenization and encoding. The Transformer part is further divided into EncoderLayers and DecoderLayers, which are each composed of small Blocks; each Block contains the multi-head attention mechanism, a feed-forward neural network, and a Norm layer. The Block design makes the model more flexible; like Lego, the number and layers of Blocks can be adjusted according to the complexity of the task and the size of the dataset.

<div align="center">
  <img src="/images/3-figures/2-2.webp" alt="图片描述" width="1600" height="1452" loading="lazy" decoding="async" />
  <p>Figure 3.8 Overall structure of the T5 model</p>
</div>

Both the Encoder and the Decoder parts of the T5 model are designed based on the Transformer architecture, mainly including two structures: Self-Attention and the feed-forward neural network. Self-Attention is used to capture the global dependencies in the input sequence, and the feed-forward neural network is used to handle the non-linear transformation of features.

Unlike the Encoder, the Decoder also contains an Encoder-Decoder Attention structure, used to capture the dependencies between the input and output sequences. These two Attention structures are almost completely identical, differing only in position encoding and the Mask mechanism. As shown in Figure 3.9, the structures of the Encoder and Decoder are as follows:

<div align='center'>
    <img src="/images/3-figures/2-3.webp" alt="Figure 3.9 Encoder and Decoder" width="1568" height="2048" loading="lazy" decoding="async" />
    <p>Figure 3.9 Encoder and Decoder</p>
</div>

T5's Self-Attention mechanism is the same as BERT's Attention mechanism; both are designed based on the Self-Attention mechanism. The Self-Attention mechanism is a global dependency modeling method that captures the global dependencies in the input sequence by computing the similarity between Query, Key, and Value. The Encoder-Decoder Attention differs only in position encoding and the Mask mechanism, mainly to distinguish the input and output sequences. As shown in Figure 3.10, the structure of Self-Attention is as follows:

<div align='center'>
    <img src="/images/3-figures/2-4.webp" alt="Figure 3.10 Self-Attention structure" width="1496" height="2240" loading="lazy" decoding="async" />
    </p>Figure 3.10 Self-Attention structure</p>
</div>

Unlike the original Transformer model, the LayerNorm of the T5 model adopts RMSNorm, which normalizes the activation values of each hidden layer by computing the Root Mean Square of each neuron. RMSNorm's parameter setup is simpler than Layer Normalization, with only one learnable parameter, and can better adapt to different tasks and datasets. The RMSNorm function can be expressed by the following mathematical formula:

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{n}\sum_{i=1}^{n}x_i^2 + \epsilon}} \cdot \gamma
$$

where:
- $x_i$ is the $i$-th element of the input vector
- $\gamma$ is a learnable scaling parameter
- $n$ is the dimensionality of the input vector
- $\epsilon$ is a small constant for numerical stability (to avoid division by zero)

This normalization helps stabilize the learning process by ensuring that the scale of the weights does not become too large or too small, which is particularly useful in deep learning models with many layers.

#### (2) Pre-training Task

The pre-training task of the T5 model is a key component; it enables the model to learn rich language representations, and the language representation ability can be transferred to various downstream tasks in subsequent fine-tuning. The dataset used for training is a large-scale text dataset containing all kinds of text data, such as Wikipedia, news, books, etc. After careful processing of the data, the 750GB dataset C4 used for training was generated and has been open-sourced in TensorflowData.

We can briefly summarize T5's pre-training task, which mainly includes the following parts:

- Pre-training task: The pre-training task of the T5 model is MLM, also known as the BERT-style objective. Specifically, 15% of the tokens in the input text are randomly masked, and the model is then asked to predict these masked tokens. This process requires no labels and can be performed on a large amount of unlabeled text.
- Input format: During pre-training, T5 converts the input text into a "text-to-text" format. For a given text sequence, some tokens are randomly selected for masking and replaced with a special placeholder (token). Then the masked token sequence is used as the model's output target.
- Pre-training dataset: T5 used its own created large-scale dataset, the "Colossal Clean Crawled Corpus" (C4), which extracted a large amount of clean English text from Common Crawl. The C4 dataset underwent a certain amount of cleaning, removing meaningless text, duplicate text, etc.
- Multi-task pre-training: T5 also tried mixing multiple tasks together for pre-training, rather than just the MLM task alone. This helps the model learn more general language representations.
- Transition from pre-training to fine-tuning: After pre-training is completed, the T5 model is fine-tuned on downstream tasks. During fine-tuning, the model is trained on task-specific datasets, and the decoding strategy is adjusted according to the task.

Through large-scale pre-training, the T5 model can learn rich language knowledge and obtain strong language representation ability, achieving excellent performance on multiple NLP tasks; pre-training is one of the key factors in T5's success.

#### (3) The Unification Idea

One core idea of the T5 model is the "unification idea": all NLP tasks can be unified as text-to-text tasks, an idea with profound influence in the field of natural language processing. Its design philosophy is to convert all different types of NLP tasks (such as text classification, translation, text generation, question answering, etc.) into a unified format: both input and output are plain text.

For example:
- For the text classification task, the input can be "classify: 这是一个很好的产品 (This is a very good product)", and the output is "正面 (positive)";
- For the translation task, the input can be "translate English to French: How are you?", and the output is "Comment ça va?".

T5 is pre-trained on large-scale text data and then fine-tuned on specific tasks. This process is similar to models such as BERT and GPT, but T5 unifies the tasks of the pre-training and fine-tuning stages into the text-to-text form, making it more adaptable across various tasks.

We can understand T5's unification idea more intuitively through Figure 3.11:

<div align='center'>
    <img src="/images/3-figures/2-0.webp" alt="Figure 3.11 T5's unification idea" width="1187" height="421" loading="lazy" decoding="async" />
    <p>Figure 3.11 T5's unification idea</p>
</div>

For different NLP tasks, a task description prefix is added before each input, explicitly specifying the type of the current task. This not only helps the model learn the common features across different tasks during the pre-training stage, but also makes it easy to quickly adapt to specific tasks during the fine-tuning stage. For example, the task prefix can be "summarize: " for summarization tasks, or "translate English to German: " for translation tasks.

T5's unification idea, by unifying all NLP tasks into the text-to-text form, simplifies the task processing pipeline and enhances the model's generality and adaptability. This idea has not only promoted the development of natural language processing technology, but also provided more convenient and efficient solutions for practical applications.

## 3.3 Decoder-Only PLM

In the previous two sections, we explained the two model architectures that evolved from the Transformer — the Encoder-Only model represented by BERT and the Encoder-Decoder model represented by T5. Naturally, one can imagine that besides these two architectures, there can be another model architecture — Decoder-Only, i.e., a model built by stacking only the Decoder.

In fact, Decoder-Only is the basic architecture of today's wildly popular LLMs; virtually all current LLMs are Decoder-Only models (except for non-Transformer architectures such as RWKV and Mamba). ChatGPT, which triggered the LLM boom, is precisely the crowning achievement of the GPT series, the representative models of the Decoder-Only line. And the LLaMA model, which currently serves as the basic architecture of open-source LLMs, was also optimized and developed on the basis of GPT's model architecture. Therefore, in this section, we will not only analyze in detail the principles, architecture, and characteristics of GPT, the representative Decoder-Only model, but also go deep into the current mainstream open-source LLMs, analyzing their structures and characteristics, and, combined with the previous analysis of other models in the Transformer series, help everyone deeply understand how today's LLMs — which carry great expectations and are considered the necessary path to AGI — evolved step by step from traditional PLMs.

First, let us study the representative model that opened the door to the world of LLMs — GPT, released by OpenAI.

### 3.3.1 GPT

GPT, i.e., Generative Pre-Training Language Model, is a pre-trained language model released by the OpenAI team in 2018. Although academia generally recognizes BERT as the representative of the pre-trained language model era, the model that first explicitly proposed the pre-training–fine-tuning idea was actually GPT. GPT proposed the concept of generic pre-training: pre-training on massive unsupervised corpora and then fine-tuning on each specific task, thereby achieving huge gains on these tasks. Although at its release it did not achieve sensational results because its performance was slightly inferior to BERT, which was released shortly after, and it failed to make the Decoder-Only architecture used by GPT the mainstream of academic research, the OpenAI team firmly chose to continuously expand the pre-training data and increase the model parameters, continuously optimizing on the GPT architecture. Ultimately, GPT-3, released in 2020, established the foundation of the LLM era, and ChatGPT, with GPT-3 as its base model, successfully opened the door of the new era, becoming the strongest competitor of the LLM era and, currently, its biggest winner.

This section will use GPT as an example to deeply analyze GPT and the Decoder-Only models it represents from three aspects: model architecture, pre-training tasks, and the development history of the GPT series, and further lead into the current mainstream LLM architecture — LLaMA.

#### (1) Model Architecture — Decoder Only

<div align='center'>
    <img src="/images/3-figures/3-0.webp" alt="Figure 3.12 GPT model structure" width="1253" height="931" loading="lazy" decoding="async" />
    <p>Figure 3.12 GPT model structure</p>
</div>

As can be seen from Figure 3.12, GPT's overall structure is somewhat similar to BERT's, except that compared with BERT's Encoder, it chose the Decoder for stacking the model structure. Since the Decoder-Only structure is also naturally suited to text generation tasks, compared with BERT, which was designed to fit NLU tasks more closely, GPT's and T5's model designs fit NLG tasks and Seq2Seq tasks better. Likewise, for a natural language text input, it is first tokenized by the tokenizer and converted into input_ids corresponding to vocabulary indices.

The input input_ids first pass through the Embedding layer, and then are positionally encoded by the Positional Embedding. Unlike BERT, which chose a trainable fully-connected layer as the positional encoding, GPT followed the Transformer's classic Sinusoidal positional encoding, i.e., absolute positional encoding via trigonometric functions; we will not repeat it here, and interested readers can refer to the analysis of the Transformer model details in Chapter 2.

After being encoded into hidden_states by the Embedding layer and the Positional Embedding layer, they can enter the Decoder. The first-generation GPT model, similar to the original Transformer model, chose 12 decoder layers; but inside the decoder layers, compared with the dual attention layer design of the original Transformer's Decoder layers, GPT's Decoder layers are actually more like the Encoder layers. Since there is no longer any encoded input from an Encoder, the Decoder layer retains only one masked attention layer, and moves the LayerNorm layer from after the attention layer (as in the Transformer) to before the attention layer. After hidden_states enter the Decoder layer, LayerNorm is applied first, then the masked attention computation; then, through a residual connection and another LayerNorm, they enter the MLP and produce the final output.

Since there is no encoded result from an Encoder, the masked attention in the Decoder layer is also a self-attention computation. That is, for an input hidden_states, three parameter matrices are used to generate query, key, and value, rather than using the Encoder's output as key and value as in the Transformer's Decoder. The subsequent attention computation process is similar to BERT's, except that after computing the attention weights, a mask matrix is used to mask the attention weights of future tokens, thereby restricting each token to attend only to the attention of tokens before it, realizing the computation of masked self-attention.

Another structural difference is that GPT's MLP layer does not choose a linear matrix for feature extraction, but rather two one-dimensional convolution kernels; however, in terms of effect, there is not much difference between the two. After passing through N Decoder layers, the hidden_states are finally mapped to the vocabulary dimension through a linear matrix, can be converted into natural language tokens, and thereby generate our target sequence.

#### (2) Pre-training Task — CLM

The Decoder-Only model structure is often more suitable for text generation tasks. Therefore, Decoder-Only models often choose the most traditional and most direct pre-training task — the Causal Language Model, hereinafter referred to as CLM.

CLM can be seen as a direct extension of the N-gram language model. The N-gram language model predicts the next token based on the previous N tokens, while CLM predicts the next token based on all previous tokens of a natural language sequence, repeating this process continuously to generate the target text sequence. In other words, CLM is a classic completion form. For example, CLM's input and output could be:

    input: 今天天气 (The weather today is)
    output: 今天天气很 (The weather today is very)

    input: 今天天气很 (The weather today is very)
    output: 今天天气很好 (The weather today is very good)

Therefore, for a task with an input target sequence length of 256 and an expected output sequence length of 256, the model will repeatedly perform 256 rounds of computation based on the first 256 tokens, 257 tokens (input + the first predicted token) ..., finally generating an output text of sequence length 512, where the first 256 tokens of this output text are the input, and the last 256 tokens are the model output we expect.

As we said earlier, the reason BERT could adopt the pre-training + fine-tuning paradigm to achieve a major breakthrough is precisely that the MLM and NSP it chose can be trained directly on massive unsupervised corpora — and clearly, CLM is an even more direct pre-training task, naturally matching the human habit of writing natural language text and directly matching downstream tasks; it is more direct than the MLM task and can be applied directly to any natural language text. Therefore, CLM can also use massive natural language corpora for large-scale pre-training.

#### (3) Development of the GPT Series Models

Since the launch of GPT-1, OpenAI has always believed in the Decoder-Only model structure and the "scale is everything" optimization approach, continuously expanding the pre-training dataset and the model scale, while making some small optimizations and corrections to the model, to keep exploring more powerful pre-trained models. From GPT-1, suppressed by BERT, to GPT-2, which did not attract enough attention, to GPT-3, which sparked emergent abilities and brought in the era of large models, and finally the epoch-making ChatGPT, OpenAI proved the correctness of its approach through years of effort.

The table below summarizes the changes in model structure and pre-training corpus size from GPT-1 to GPT-3:

Model | Decoder Layer | Hidden_size | Number of Attention Heads | Attention Dimension | Total Parameters | Pre-training Corpus
---- | --------------|------------|------------|----------|----------|----------
GPT-1|12|3072|12|768|0.12B|5GB
GPT-2|48|6400|25|1600|1.5B|40GB
GPT-3|96|49152|96|12288|175B|570GB

GPT-1 is the pioneering work of the GPT series and the first pre-trained model to use Decoder-Only. However, GPT-1's model scale and pre-training data were both small; it inherited the traditional Transformer's model structure, using 12 Decoder Blocks and a hidden layer dimension of 768, with a model parameter count of only 117 million (0.12B), pre-trained on the 5GB BooksCorpus dataset. It can be seen that GPT-1's parameter scale and pre-training scale are roughly comparable to BERT-base, but its performance is somewhat inferior to BERT-base, which is also the reason the GPT series of models failed to become the representative of the pre-trained language model era.

GPT-2 was OpenAI's product of further exploring the multi-task learning capability of pre-trained language models on the basis of GPT-1. GPT-2's model structure is roughly comparable to GPT-1's, except that the model parameter scale was enlarged and Post-Norm was changed to Pre-Norm (i.e., performing the LayerNorm computation first, then entering the attention layer computation). The core reason for these changes is that, as the number of model layers and the scale increased, the risk of gradient vanishing and exploding kept growing; to make the model's gradients more stable, the above structures were optimized.

GPT-2's core improvement was a substantial increase in the pre-training dataset and the model scale. GPT-2's Decoder Block layers reached 48 (note: GPT-2 released four specifications of models; here we refer only to the largest specification, the GPT-2 model), the hidden layer dimension reached 1600, and the model's total parameter count reached 1.5 billion (1.5B); it was pre-trained using its own crawled 40GB WebText dataset, surpassing the 1st generation by an order of magnitude in both model structure and pre-training size.

GPT-2's other major breakthrough was taking zero-shot (zero-shot learning) as the main goal, i.e., not fine-tuning the model at all and directly asking the model to solve tasks. For example, in the traditional pre-training–fine-tuning paradigm, to solve a problem we generally need to collect hundreds or thousands of training samples and fine-tune a pre-trained language model on these samples to solve the problem. Zero-shot, by contrast, emphasizes using no training samples at all, and solving the problem directly by describing the problem to the pre-trained language model. The zero-shot approach is naturally a further step beyond the pre-training–fine-tuning paradigm — a more efficient natural language paradigm — but in GPT-2's era, the model's capability was not yet sufficient to support good zero-shot results; it was only in the era of large models that zero-shot and its extension, few-shot (few-shot learning), gradually became mainstream.

GPT-3 further demonstrated OpenAI's core idea of "brute force over brains" and is the pioneering work of LLMs. On the basis of GPT-2, OpenAI further increased the model scale and the amount of pre-training data, with total parameters reaching 175B — a truly deserved "large language model". In terms of model structure, there were basically no major improvements, only that due to the enormous model scale, sparse attention was used to replace the traditional attention mechanism. For pre-training data, sampling was done separately from large corpora such as CC, WebText, and Wikipedia, totaling 45T sampled, 570GB after cleaning. According to estimates, GPT-3 required training on a distributed cluster of 1024 A100s (80GB of memory) for 1 month.

The reason GPT-3 is called the pioneering work of LLMs, besides its enormous scale highlighting emergent abilities, lies in the important idea it proposed: few-shot. Few-shot is an improvement on zero-shot. Researchers found that even for GPT-3 at 175B, achieving good zero-shot performance remained quite difficult. Few-shot is a compromise on zero-shot, aiming to provide the model with a small number of examples to teach it to complete the task. Few-shot generally adds 3–5 examples to the prompt (i.e., the model's input) to help the model understand. For example, for the sentiment classification task:

    zero-shot: 请你判断'这真是一个绝佳的机会'的情感是正向还是负向，如果是正向，输出1；否则输出0 (Please judge whether the sentiment of '这真是一个绝佳的机会' (This is truly a great opportunity) is positive or negative; if positive, output 1; otherwise output 0)

    few-shot: 请你判断'这真是一个绝佳的机会'的情感是正向还是负向，如果是正向，输出1；否则输出0。你可以参考以下示例来判断：'你的表现非常好'——1；'太糟糕了'——0；'真是一个好主意'——1。(Please judge whether the sentiment of '这真是一个绝佳的机会' (This is truly a great opportunity) is positive or negative; if positive, output 1; otherwise output 0. You can refer to the following examples to make the judgment: '你的表现非常好' (Your performance is very good) — 1; '太糟糕了' (It's terrible) — 0; '真是一个好主意' (What a great idea) — 1.)

By providing the model with a small number of examples, the model can achieve performance far better than zero-shot. Few-shot is also known as In-context Learning, i.e., letting the model learn the solution to the problem from the examples in the provided context. The powerful ability GPT-3 demonstrated on few-shot brought important progress to breakthroughs in NLP. If for the vast majority of tasks the model can be made to solve them by manually constructing 3–5 examples, its efficiency will be far higher than the traditional pre-training–fine-tuning paradigm, making further real-world application of NLP possible — and this is precisely the core advantage of LLMs.

On the basis of the GPT series of models, by introducing the three-stage training of pre-training–instruction fine-tuning–reinforcement learning from human feedback, OpenAI released the epoch-making ChatGPT, triggering the boom of large models. It was also on the basis of GPT-3 and ChatGPT that the releases of models such as LLaMA and ChatGLM further revealed the endless potential of LLMs. In the next section, we will deeply dissect the current universal architecture of LLMs — LLaMA.
### 3.3.2 LLaMA

The LLaMA models are a series of large pre-trained language models developed by Meta (formerly Facebook). From LLaMA-1 to LLaMA-3, the LLaMA series of models has demonstrated the evolution of large-scale pre-trained language models and their significant potential in practical applications.

#### (1) Model Architecture — Decoder Only

Like the GPT series of models, the LLaMA models are also pre-trained language models based on the Decoder-Only architecture. The overall structure of the LLaMA models is similar to that of the GPT series, differing only in model scale and pre-training datasets. Figure 3.13 shows a schematic diagram of the LLaMA model's architecture:

<div align='center'>
    <img src="/images/3-figures/3-1.webp" alt="Figure 3.13 LLaMA-3 model structure" width="1600" height="1142" loading="lazy" decoding="async" />
    <p>Figure 3.13 LLaMA-3 model structure</p>
</div>

Like GPT, LLaMA's processing pipeline also starts with encoding the input text through the tokenizer, converting it into a series of input_ids. These input_ids are the data format the model can understand and process. Next, these input_ids go through the transformation of the embedding layer, where each input_id is mapped to a vector in a high-dimensional space, i.e., a word vector. Meanwhile, the position information of the input text is also encoded through the positional embedding layer, ensuring the model can understand the word-order context information.

In this way, the input_ids, through the combination of the embedding layer and the positional embedding layer, form the hidden_states. The hidden_states contain the semantic and position information of the input text and are the basis for the model's subsequent processing; the hidden_states are then fed into the model's decoder layers.

In the decoder layers, the hidden_states undergo a series of processing operations, which are composed of multiple decoder blocks. Each decoder block is a core component of the model, responsible for deeply analyzing and transforming the hidden_states. Inside each decoder block, there is first a masked self-attention layer. In this layer, the model computes the three vectors query, key, and value separately. These vectors are obtained through linear transformation of the hidden_states and are the basis for computing the attention weights. Then the softmax function is used to compute the attention score, which reflects the strength of the association between different positions. Through the attention score, the model can determine, when generating the current word, how much attention should be paid to the hidden_states at different positions. Then the model multiplies the value vector with the attention score to obtain the weighted value; this is the result of the attention.

After the masked self-attention layer is completed, the hidden_states enter the MLP layer. In this multi-layer perceptron layer, the model performs further feature extraction on the hidden_states through two fully-connected layers. The first fully-connected layer maps the hidden_states to an intermediate dimension, and then a non-linear transformation is performed through an activation function, increasing the model's non-linear capability. The second fully-connected layer maps the features back to the original hidden_states dimension.

Finally, after being processed by multiple decoder blocks, the hidden_states go through a linear layer for the final mapping, and the output dimension of this linear layer is the same as the vocabulary dimension. In this way, the model can generate the probability distribution of the target sequence according to the hidden_states, and then, through methods such as sampling or greedy decoding, generate the final output sequence. This process demonstrates the powerful sequence generation capability of the LLaMA model.

#### (2) Development History of the LLaMA Models

**LLaMA-1 series**:

- Meta released LLaMA-1 in February 2023, including four versions with 7B, 13B, 30B, and 65B parameters.
- These models were pre-trained on a corpus of more than 1T tokens; the largest 65B-parameter model was trained on 2,048 A100 80G GPUs for nearly 21 days.
- Thanks to its open-source nature and excellent performance, LLaMA-1 quickly became one of the most popular large models in the open-source community.

**LLaMA-2 series**:

- In July 2023, Meta released LLaMA-2, including four versions with 7B, 13B, 34B, and 70B parameters; all except the 34B model have been open-sourced.
- LLaMA-2 expanded the pre-training corpus to 2T tokens and doubled the model's context length from 2,048 to 4,096.
- It introduced techniques such as Grouped-Query Attention (GQA).

**LLaMA-3 series**:

- In April 2024, Meta released LLaMA-3, including two versions with 8B and 70B parameters, while revealing that the 400B LLaMA-3 was still in training.
- LLaMA-3 supports 8K long text and adopts a tokenizer with higher encoding efficiency, with a vocabulary size of 128K.
- It used a pre-training corpus of more than 15T tokens, more than 7 times that of LLaMA-2.

The LLaMA models are renowned for their technical innovation, multiple parameter versions, large-scale pre-training, and efficient architecture design. The models support parameter counts ranging from 7 billion to tens of billions, adapting to application needs of different scales. LLaMA-1 quickly gained community popularity for its open-source nature and excellent performance, while LLaMA-2 and LLaMA-3 further improved model performance and application scope by introducing Grouped-Query Attention and supporting longer text inputs. In particular, LLaMA-3, by adopting an efficient tokenizer with a 128K vocabulary size and a massive training data of 15T tokens, achieved major progress in multilingual and multi-task processing. Meta's continued attention to model safety and community support suggests that LLaMA will continue to serve as an important driving force for the development of AI technology, promoting technology application and innovation worldwide.

### 3.3.3 GLM

The GLM series of models, developed by Zhipu, is one of the mainstream Chinese LLMs, including the ChatGLM1, 2, 3 and GLM-4 series of models, covering a variety of application scenarios such as instruction understanding and code generation, and has achieved SOTA performance on multiple Chinese evaluation sets.

ChatGLM-6B is the pioneering work of the GLM series, the earliest open-source Chinese LLM in China in 2023, and also the earliest LLM to propose a unique model architecture different from GPT and LLaMA. Throughout the development history of Chinese LLMs, GLM has a unique and major technical significance. This section will briefly describe the development of the GLM series and introduce its unique technical ideas, different from the GPT and LLaMA series of models.

#### (1) Model Architecture — Slight Modifications Relative to GPT

GLM was originally a general language model base proposed by the Department of Computer Science of Tsinghua University. Its core idea is to add the MLM idea on the basis of the traditional CLM pre-training task, thereby building a unified model that performs well on both NLG and NLU tasks.

In terms of overall model structure, GLM is roughly similar to GPT, both being Decoder-Only structures, with only three subtle differences:

1. It uses Post Norm rather than Pre Norm. Post Norm means that when computing the residual connection, the residual computation is completed first, and then the LayerNorm computation is performed; whereas models such as GPT and LLaMA all use Pre Norm, i.e., performing the LayerNorm computation first and then the residual computation. Relatively speaking, Post Norm, because it normalizes after the residual, has a stronger regularization effect on the parameters, and thus the model's robustness is better; in Pre Norm, since a portion of the parameters is directly added at the end, this portion of parameters does not need regularization, which precisely can prevent the model's gradient explosion or gradient vanishing. Therefore, for larger-scale models, Pre Norm is generally considered to work better. But the GLM paper proposed that using Post Norm can avoid numerical errors in LLMs (although mainstream LLMs still use Pre Norm);

2. It uses a single linear layer to implement the final token prediction, rather than using an MLP; this structure is simpler and more robust, i.e., it reduces the parameter count of the final output, placing a larger parameter count in the model itself;

3. The activation function was changed from ReLU to GeLUs. ReLU is the traditional activation function, whose core computational logic is to remove propagation of values smaller than 0 and keep propagation of values larger than 0; the core of GeLUs is to apply a non-linear mapping to forward propagation values close to 0, ensuring a non-linear output after the activation function with a certain degree of continuity.

#### (2) Pre-training Task — GLM

GLM's core innovation mainly lies in the GLM (General Language Model) task it proposed, which is also the origin of GLM's name. GLM is a pre-training method combining the autoencoding idea and the autoregressive idea. The so-called autoencoding idea is actually the task-learning approach of MLM: randomly delete contiguous tokens from the input text and ask the model to learn the deleted tokens; the so-called autoregressive idea is actually the traditional CLM task-learning approach, i.e., requiring the model to reconstruct the contiguous tokens in order.

GLM realizes the combination of the MLM and CLM ideas by optimizing an autoregressive blank-filling task. Its core idea is: for an input sequence, random masking is performed similar to MLM, but what is masked is not single tokens as in MLM, but a contiguous span of tokens each time; during learning, the model needs to use the context around the masked span to predict the masked span, and within the masked span it also needs to complete the prediction of the masked tokens in the CLM manner. For example, the input and output might be:

    Input: I <MASK> because you <MASK>
    Output: <MASK> - love you; <MASK> - are a wonderful person

By combining the MLM and CLM ideas, it both fits generation-type tasks that produce token by token, and forces the model to learn the implicit relations of the input text from both front and back directions, thereby fitting understanding-type tasks. The GLM model produced with the GLM pre-training task demonstrated, to a certain extent, superior performance beyond BERT-family models of the same scale:

<div align='center'>
    <img src="/images/3-figures/3-2.webp" alt="Figure 3.14 alt text" width="892" height="580" loading="lazy" decoding="async" />
    <p>Figure 3.14 alt text</p>
</div>

However, the advantages of the GLM pre-training task were mostly displayed in the pre-trained model era. After entering the LLM era, for pre-training at super-large scale and volume, CLM showed advantages far beyond MLM. By increasing the model scale and expanding the pre-training scale, generative models pre-trained with CLM can also surpass MLM-trained understanding models in text understanding. Therefore, the ChatGLM series of models only used the GLM pre-training idea in the first-generation model; starting from ChatGLM2, it returned to the traditional CLM modeling. Although, from the perspective of the overall development path of LLMs, the GLM pre-training task seems to be a failed attempt, its idea of fusing CLM and MLM through ingenious design, and of producing a Chinese open-source native LLM at the earliest opportunity, still has considerable reference significance.

#### (3) Development of the GLM Family

On the basis of the GLM model (i.e., the early pre-trained model using the native GLM architecture and pre-training task), and following ChatGPT's technical approach with SFT and RLHF, Zhipu released the first Chinese open-source LLM, ChatGLM-6B, in March 2023, which became the starting point for many Chinese LLM researchers. ChatGLM-6B was pre-trained on 1T of corpus and supports a context length of 2K.

In June 2023, Zhipu open-sourced ChatGLM2-6B. Compared with the first generation, ChatGLM2 extended the context length to 32K and achieved a substantial breakthrough in model performance through a larger pre-training scale. However, in ChatGLM2, the model architecture basically returned to the LLaMA architecture, introducing the MQA attention mechanism, and the pre-training task also returned to the classic CLM, abandoning GLM's failed attempt.

ChatGLM3-6B was released in October 2023; compared with the second generation, it achieved the then-SOTA performance in semantics, mathematics, reasoning, code, and knowledge, but the official technical report stated that ChatGLM3 had no changes in model architecture relative to the second generation, and that the main optimizations came from a more diverse training dataset, more sufficient training steps, and more optimized training strategies. Another important improvement of ChatGLM3 is that it began to support function calling and a code interpreter; developers can directly use the open-source ChatGLM3 to develop Agents, giving it broader application value.

In January 2024, Zhipu released the GLM-4 series of models supporting a 128K context, including multiple types, claiming in evaluation that it reached the level of GPT-4 on English benchmarks. However, Zhipu did not open-source GLM-4 directly, but open-sourced its lightweight version, the GLM-4-9B model, which was pre-trained on a 1T-token multilingual corpus, has a context length of 8K, and uses the same pipeline and data as GLM-4 for post-training. With less training computation, it surpassed Llama-3-8B and supports the functions of all tools in GLM-4.

Figure 3.15 shows the evolution of the performance of the GLM series of models on benchmark sets:

<div align='center'>
    <img src="/images/3-figures/3-3.webp" alt="Figure 3.15 alt text" width="1019" height="728" loading="lazy" decoding="async" />
    <p>Figure 3.15 alt text</p>
</div>

**References**

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
