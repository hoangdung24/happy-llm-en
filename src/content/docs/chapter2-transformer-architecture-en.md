---
title: "Chapter 2: Transformer Architecture"
description: "The Transformer architecture explained: self-attention, multi-head attention, encoder and decoder blocks, and positional encoding."
---

## 2.1 The Attention Mechanism

### 2.1.1 What is the Attention Mechanism

NLP advances from statistical machine learning toward deep learning. Text representation is the core problem of NLP. It also advanced from statistical learning to deep learning. Chapter 1 introduced this change. Text representation evolved from early vector space models and language models. These models used statistical learning. It then moved to the single-layer neural network of Word2Vec. It then entered the era of learning text representations with neural networks. Neural networks came from Computer Vision (CV). They have three core architectures:

- Feedforward Neural Network (FNN). In this network, data flows in one direction from the input layer to the output layer. It has no recurrent structure. Layers pass information to each other through full connections or specific schemes. The Multi-Layer Perceptron (MLP) is the most common form. In the MLP, each neuron of a layer connects fully to every neuron of the layers above and below. Figure 2.1 shows this:

<div align="center">
  <img src="/images/2-figures/1-0.webp" alt="Figure 2.1 Fully Connected Neural Network" width="1280" height="547" loading="lazy" decoding="async" />
  <p>Figure 2.1 Fully Connected Neural Network</p>
</div>

- Convolutional Neural Network (CNN). This network uses convolutional layers. Their number of trainable parameters is much smaller than that of a fully connected neural network. The layers extract features and learn them. Figure 2.2 shows this:

<div align="center">
  <img src="/images/2-figures/1-1.webp" alt="Figure 2.2 Convolutional Neural Network" width="1280" height="489" loading="lazy" decoding="async" />
  <p>Figure 2.2 Convolutional Neural Network</p>
</div>

- Recurrent Neural Network (RNN). This network can use historical information as input. It contains loops and self-repetition. Figure 2.3 shows this:

<div align="center">
  <img src="/images/2-figures/1-2.webp" alt="Figure 2.3 Recurrent Neural Network" width="1280" height="809" loading="lazy" decoding="async" />
  <p>Figure 2.3 Recurrent Neural Network</p>
</div>

NLP tasks usually process sequential text. RNNs are specialized for sequential and time-series data. They often give the best results on NLP tasks. Before the attention mechanism appeared, RNNs and their derivative architecture LSTM ruled the NLP field. Chapter 1 discussed ELMo. It was the text representation model that started the idea of pre-training. It uses a bidirectional LSTM as its network architecture.

RNNs and LSTMs have two advantages. They capture temporal information. They also suit sequence generation well. But they have two flaws that are hard to fix:

1. The sequential computation pattern models temporal information well. But it limits the parallel computing capability of computers. A computer must input and compute sequences in order. This greatly restricts the parallel computing capability of Graphics Processing Units (GPUs). Models built on RNN architectures have a moderate parameter count. But they have high computational time costs.

2. RNNs struggle to capture correlations over long sequences. In the RNN architecture, two inputs that sit far apart are harder to relate. An RNN must also read the entire sequence into memory and compute it in order. This limits sequence length. LSTMs optimize this to some extent through gating mechanisms. But RNNs remain weak at capturing long-distance correlations.

To solve these problems, Vaswani et al. drew on the attention mechanism. The CV field proposed this mechanism, and researchers often integrated it into RNNs. Note that the attention mechanism flourished in NLP. But the CV field did propose it first. The authors built a new neural network from attention mechanisms only: the Transformer. This network is the ancestor and core architecture of the Large Language Model (LLM). It made the attention mechanism one of the most central architectures in deep learning.

So what exactly is the attention mechanism?

The attention mechanism originated in the computer vision field. Its core idea is simple. When we look at an image, we often do not need to see all of its content. We can focus our attention on the key parts. In natural language processing, we can get more efficient, higher-quality computation. We do this by focusing attention on one or a few tokens.

The attention mechanism has three core variables: Query (the query value), Key (the key value), and Value (the true value). An example shows what each variable represents. Suppose we have a news report. We want to find the time of the event. Our Query can be a vector that resembles "time" or "date". We use text here for ease of understanding. But it is actually a dense vector. The Key and the Value are the entire text. We compute on the Query and the Key. This gives a weight. Starting from the Query, the weight reflects the relative magnitude of attention that we must distribute over each token of the text. We then compute on the weight and the Value. The final result is the attention of the entire text, computed from the Query.

In detail, the attention mechanism computes the correlation between Query and Key. It uses this correlation to form a weighted sum of the true values. This fits the correlation between each word in the sequence and every other word.

### 2.1.2 A Deeper Understanding of the Attention Mechanism

We just said that the attention mechanism has three core variables: the query value Query, the key value Key, and the true value Value. We now use a dictionary as an example. We analyze step by step how the attention computation formula is derived. This helps readers understand the attention mechanism deeply. First, consider the following dictionary:

```json
{
    "apple":10,
    "banana":5,
    "chair":2
}
```

Here, the dictionary's keys are the Keys of the attention mechanism. The dictionary's values are the Values. A dictionary supports exact string matching. For example, the value we want to look up (the query value Query) is "apple". We can match the Query directly against the Key. This gives the corresponding Value.

But what if the Query we want to match is a concept that contains multiple Keys? For example, we want to look up "fruit". In that case, we must match both apple and banana. We must not match chair. So we usually combine the Values that correspond to the Keys. This gives the final Value.

For example, when our Query is "fruit", we can assign the following weights to the three Keys:

```json
{
    "apple":0.6,
    "banana":0.4,
    "chair":0
}
```

Then the value we finally retrieve must be:

$$
value = 0.6 * 10 + 0.4 * 5 + 0 * 2 = 8
$$

The different weights assigned to different Keys are what we call attention scores. They show how much attention we must give to each Key to answer the Query. But how do we compute the attention scores for each Query? Intuitively, a Key that is more relevant to the Query must get a larger attention weight. But how can we find a reasonable method that computes correct attention scores?

Chapter 1 mentioned the concept of word vectors. Proper training and fitting let word vectors represent semantic information. Semantically similar words then sit closer together in vector space. Semantically distant words sit farther apart. We usually use Euclidean distance to measure the similarity of word vectors. But we can also measure it with the dot product:

$$
v·w = \sum_{i}v_iw_i
$$

By the definition of word vectors, the dot product of two semantically similar word vectors must be greater than 0. The dot product of semantically dissimilar word vectors must be less than 0.

So we can use the dot product to compute the similarity between words. Suppose our Query is "fruit" with corresponding word vector $q$. Our Keys correspond to the word vectors $k = [v_{apple} v_{banana} v_{chair}]$. We can then compute the similarity of the Query with each key:

$$
x = qK^T
$$

Here, K is the matrix formed by stacking the word vectors of all the Keys. By the definition of matrix multiplication, x is the dot product of q with each k. The x we now get reflects the similarity between the Query and each Key. We then pass it through a Softmax layer. This converts it into weights that sum to 1:

$$
\text{softmax}(x)_i = \frac{e^{xi}}{\sum_{j}e^{x_j}}
$$

The resulting vector reflects the similarity between the Query and each Key. At the same time, the weights sum to 1. These are our attention scores. Finally, we multiply the attention scores by the value vectors. From the process above, we can get the basic formula of attention computation:

$$
attention(Q,K,V) = softmax(qK^T)v
$$

But the value is still a scalar at this point. In this pass we queried only one Query. We can convert the value into a vector of dimension $d_v$. We can also query multiple Queries at once. We do this by stacking the word vectors of the multiple Queries into a matrix Q. This yields the formula:

$$
attention(Q,K,V) = softmax(QK^T)V
$$

We are now one final step away from the standard attention formula. In the previous formula, Q and K have a relatively large dimension $d_k$. The softmax scaling is then very easily affected. This makes the differences between values large. This affects gradient stability. So we scale the product of Q and K:

$$
attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
$$

This is the core computation formula of the attention mechanism.

### 2.1.3 Implementing the Attention Mechanism

Based on the above, we can easily implement the attention mechanism in PyTorch:

```python
'''Attention computation function'''
def attention(query, key, value, dropout=None):
    '''
    args:
    query: query matrix
    key: key matrix
    value: value matrix
    '''
    # Get the dimension of the key vectors; the key and value vectors share the same dimension
    d_k = query.size(-1) 
    # Compute the inner product of Q and K, divided by the square root of dk
    # transpose——equivalent to transposing
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # Softmax
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
        # sampling
     # Compute the weighted sum of value based on the computed results
    return torch.matmul(p_attn, value), p_attn

```

Keep one point in mind about the code above. We assume the inputs q, k, v are word vector matrices that have already been transformed, i.e., the Q, K, V in the formula. With just these few lines of code, we can implement the core attention computation.

### 2.1.4 Self-Attention

From the analysis above, we can see the essence of the attention mechanism. It computes the similarity between the elements of two sequences in turn. It finds the relevance of each element of one sequence to each element of the other. It then weights by that relevance, that is, it allocates attention. These two sequences are the sources of Q, K, V in our computation.

In practical applications, we often need only the attention result between the Query and the Key. An additional true value Value rarely exists. That is, we actually need to fit only two text sequences. In the classic attention mechanism, Q often comes from one sequence. K and V come from another sequence. Both pass through parameter matrices. This fits the relationship between the two sequences. Consider the Transformer's Decoder structure. Q comes from the Decoder's input. K and V come from the Encoder's output. This fits the relationship between the encoded information and the historical information. This helps to combine both kinds of information and to predict the future.

But the Transformer's Encoder structure uses a variant of the attention mechanism: self-attention. Self-attention computes the attention distribution of each element of the sequence over the other elements. That is, in the computation, Q, K, V all come from the same input. They pass through different parameter matrices. In the Encoder, Q, K, V come from multiplying the input with the parameter matrices $W_q$, $W_k$, $W_v$. This fits the relationship of each token in the input sentence to all other tokens.

The self-attention mechanism lets us find the magnitude of the correlation between each token in a text and all other tokens. This models the dependencies within the text. In code, the self-attention mechanism actually passes the same parameter as the input to Q, K, V:

```python
# attention is the attention computation function defined above
attention(x, x, x)
```

### 2.1.5 Masked Self-Attention

Masked self-attention, i.e., Mask Self-Attention, is the self-attention mechanism that uses an attention mask. The mask hides tokens at certain positions. During learning, the model ignores the masked tokens.

An attention mask has one core purpose. It lets the model use only historical information for prediction, without seeing future information. Transformer models that use the attention mechanism also learn through a language-modeling task similar to n-gram. The model gets a text sequence. It predicts the next token from the previous tokens. It repeats this until it completes the entire text sequence.

For example, the text sequence to be learned is 【BOS】I like you【EOS】. The model predicts and learns in the following order:

    Step 1: input 【BOS】, output I
    Step 2: input 【BOS】I, output like
    Step 3: input 【BOS】I like, output you
    Step 4: input 【BOS】I like you, output 【EOS】

In theory, the model needs a large enough training corpus. With the process above, the model can then learn to model any kind of text sequence. That is, it can complete any text.

But we can see that the process above is serial. Step 1 must finish before Step 2 can run. This continues until the model completes the entire sequence. At the very beginning we said that one of the Transformer's core advantages over the RNN is parallel computation. Parallel computation gives higher computational efficiency. Suppose the model must run the above process serially for each training sample to finish learning. Then it clearly does not achieve parallel computation, and computational efficiency is low.

To solve this problem, the Transformer proposed the masked self-attention method. Masked self-attention generates a mask that hides future information. For example, the text sequence to be learned is still 【BOS】I like you【EOS】. We use 【MASK】 as the attention mask. The model's input is then:

    <BOS> 【MASK】【MASK】【MASK】【MASK】
    <BOS>    I   【MASK】 【MASK】【MASK】
    <BOS>    I     like  【MASK】【MASK】
    <BOS>    I     like    you  【MASK】
    <BOS>    I     like    you   </EOS>

In each row of input, the model still sees only the preceding tokens. It predicts the next token. But keep in mind that these inputs are no longer a serial process. We can feed them into the model in parallel all at once. The model needs only each sample to predict the next token from the unmasked tokens. This realizes a parallel language model.

We can observe the mask above. It is actually an upper-triangular matrix with the same length as the text sequence. We can create an upper-triangular matrix of the same length as the input. This matrix serves as the attention mask. We then use the mask to mask out parts of the input. The input dimension is (batch_size, seq_len, hidden_size). Our Mask matrix generally has dimension (1, seq_len, seq_len). Broadcasting enables the computation for different samples in the same batch.

In the concrete implementation, we generate the Mask matrix with the following code:

```python
# Create an upper-triangular matrix to hide future information.
# First create a 1 * seq_len * seq_len matrix with the full function
mask = torch.full((1, args.max_seq_len, args.max_seq_len), float("-inf"))
# The triu function creates an upper-triangular matrix
mask = torch.triu(mask, diagonal=1)
```

The generated Mask matrix is an upper-triangular matrix. The elements in the upper-triangular positions are all -inf. The elements elsewhere are set to 0.

During attention computation, we add the computed attention scores to this mask. We then perform the Softmax operation:

```python
# Here scores are the computed attention scores, and mask is the mask matrix generated above
scores = scores + mask[:, :seqlen, :seqlen]
scores = F.softmax(scores.float(), dim=-1).type_as(xq)
```

After the addition, the attention scores in the upper-triangular region all become `-inf`. This region corresponds to the positions of the tokens that we must mask. The scores in the lower-triangular region do not change. After the Softmax operation, the `-inf` values become 0. The model then ignores the attention scores computed for the upper-triangular region. This realizes attention masking.

### 2.1.6 Multi-Head Attention

The attention mechanism can achieve parallelization. It can also fit long-term dependencies. But a single attention computation can fit only one kind of correlation. A single attention mechanism is weak at fitting the correlations in a sentence sequence. So the Transformer uses the Multi-Head Attention mechanism. It performs multiple attention computations on the same corpus at the same time. Each attention computation fits a different relation. It then concatenates the multiple final results as the final output. This fits the linguistic information more deeply.

In the original paper, the authors also confirmed one point by experiment. In multi-head attention computation, each different attention head can fit different information in the sentence. Figure 2.4 shows this:

<div align="center">
  <img src="/images/2-figures/1-3.webp" alt="Figure 2.4 Multi-Head Attention Mechanism" width="586" height="597" loading="lazy" decoding="async" />
  <p>Figure 2.4 Multi-Head Attention Mechanism</p>
</div>

The upper and lower parts show the results of two attention heads. Both heads compute self-attention on the same sentence sequence. We can see that different attention heads fit different levels of correlated information. Multiple attention heads compute at the same time. They fit the sentence's relations more deeply.

In fact, the so-called multi-head attention mechanism does this. It performs multiple groups of self-attention processing on the original input sequence. It then concatenates the self-attention results of each group. It passes them through a linear layer to get the final output. We can express it formally as:

$$
\mathrm{MultiHead}(Q, K, V) = \mathrm{Concat}(\mathrm{head_1}, ...,
\mathrm{head_h})W^O    \\
    \text{where}~\mathrm{head_i} = \mathrm{Attention}(QW^Q_i, KW^K_i, VW^V_i)
$$

Its most straightforward code implementation is not complicated. n heads means n groups of 3 parameter matrices. Each group performs the same attention computation. But the parameter matrices are different. Backpropagation therefore yields different attention results. We then concatenate the n results and output them.

But the implementation above has high time and space complexity. We can implement parallel multi-head computation through matrix operations. The core logic uses three combined matrices. They replace the combination of n parameter matrices. Inner product then concatenation is actually equivalent to concatenation then inner product. For the concrete implementation, refer to the following code:

```python
import torch.nn as nn
import torch

'''Multi-head self-attention computation module'''
class MultiHeadAttention(nn.Module):

    def __init__(self, args: ModelArgs, is_causal=False):
        # Constructor
        # args: configuration object
        super().__init__()
        # The hidden layer dimension must be an integer multiple of the number of heads, because later we will split the input into as many matrices as there are heads
        assert args.dim % args.n_heads == 0
        # The dimension of each head, equal to the model dimension divided by the total number of heads.
        self.head_dim = args.dim // args.n_heads
        self.n_heads = args.n_heads

        # Wq, Wk, Wv parameter matrices, each parameter matrix being n_embd x dim
        # Here, three combined matrices replace the combination of n parameter matrices; the logic is that inner product then concatenation
        # is actually equivalent to concatenation then inner product.
        # Readers who do not understand this can simulate it themselves: each linear layer is effectively the concatenation of n parameter matrices
        self.wq = nn.Linear(args.n_embd, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.n_embd, self.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.n_embd, self.n_heads * self.head_dim, bias=False)
        # Output weight matrix, of dimension dim x dim (head_dim = dim / n_heads)
        self.wo = nn.Linear(self.n_heads * self.head_dim, args.dim, bias=False)
        # Dropout for attention
        self.attn_dropout = nn.Dropout(args.dropout)
        # Dropout for residual connections
        self.resid_dropout = nn.Dropout(args.dropout)
        self.is_causal = is_causal

        # Create an upper-triangular matrix to hide future information
        # Note: since this is multi-head attention, the Mask matrix has one more dimension than the one we defined earlier
        if is_causal:
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            # Register as a model buffer
            self.register_buffer("mask", mask)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):

        # Get batch size and sequence length, [batch_size, seq_len, dim]
        bsz, seqlen, _ = q.shape

        # Compute query (Q), key (K), value (V): the inputs pass through the parameter-matrix layers, dimension (B, T, n_embed) x (n_embed, dim) -> (B, T, dim)
        xq, xk, xv = self.wq(q), self.wk(k), self.wv(v)

        # Split Q, K, V into multiple heads, dimension (B, T, n_head, dim // n_head), then swap dimensions to become (B, n_head, T, dim // n_head)
        # Because in the attention computation we take the last two dimensions to participate in the computation
        # Why expand by B*T*n_head*C//n_head and then swap dimensions 1 and 2, rather than expanding directly by the attention input: view's expansion
        # simply lays out the entire input, then constructs it as required; you can see that only the above operation achieves our goal of extracting
        # the corresponding part of each head
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Attention computation
        # Compute QK^T / sqrt(d_k), dimension (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        # Masked self-attention must have an attention mask
        if self.is_causal:
            assert hasattr(self, 'mask')
            # Truncate to the sequence length here, since some sequences may be shorter than max_seq_len
            scores = scores + self.mask[:, :, :seqlen, :seqlen]
        # Compute softmax, dimension (B, nh, T, T)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        # Apply Dropout
        scores = self.attn_dropout(scores)
        # V * Score, dimension (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        output = torch.matmul(scores, xv)

        # Restore the time dimension and merge the heads.
        # Concatenate the multi-head results: first transpose to (B, T, n_head, dim // n_head), then reshape to (B, T, n_head * dim // n_head)
        # The contiguous function is used to allocate a new block of memory for storage, because PyTorch raises an error when setting transpose before view,
        # since view is derived directly from the underlying storage, yet transpose does not change the underlying storage, so additional storage is needed
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # Finally, project back into the residual stream.
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output
```

## 2.2 Encoder-Decoder

The previous section introduced the core of the Transformer, the attention mechanism. In the paper "Attention is All You Need", the authors build the Transformer model with only the attention mechanism. They drop the traditional RNN and CNN architectures. This change brings a great transformation to the NLP field. In the Transformer, two core components use the attention mechanism. These components are the Encoder (encoder) and the Decoder (decoder). In fact, later pre-trained language models derive from the Transformer architecture. Most of them improve the Encoder-Decoder part to build new model architectures. Examples are BERT, which uses only the Encoder, and GPT, which uses only the Decoder.

This section starts from the Seq2Seq task that the Transformer targets. It builds on the attention mechanism from the previous section. It analyzes the Encoder-Decoder structure of the Transformer.

### 2.2.1 Seq2Seq Models

Seq2Seq, that is, sequence-to-sequence, is a classic NLP task. The model inputs a natural language sequence $input = (x_1, x_2, x_3...x_n)$. It outputs a natural language sequence of possibly different length $output = (y_1, y_2, y_3...y_m)$. In fact, Seq2Seq is the most classic NLP task. Almost all NLP tasks can be viewed as Seq2Seq tasks. For example, a text classification task outputs a target sequence of length 1 (that is, $m$ = 1 in the formula above). A part-of-speech tagging task outputs a target sequence of the same length as the input sequence (that is, $m$ = $n$ in the formula above).

The machine translation task is a classic Seq2Seq task. For example, our input might be “今天天气真好” (The weather is really nice today) and the output "Today is a good day.". The Transformer is a classic Seq2Seq model. The model takes one text sequence as input and emits another as output. In fact, the Transformer was first applied to the machine translation task.

For Seq2Seq tasks, the general approach is to encode the natural language sequence and then decode it. Encoding passes the input natural language sequence through hidden layers into a vector (or matrix) that represents semantics. You can understand this vector as a more complex word-vector representation. Decoding passes the vector or matrix obtained by encoding through hidden layers. Then it decodes the result into the corresponding natural language target sequence. Encoding and then decoding can complete the Seq2Seq task.

The Encoder in the Transformer performs the encoding process above. The Decoder performs the decoding process. Figure 2.5 shows the Transformer structure:

<div align="center">
  <img src="/images/2-figures/2-0.webp" alt="Figure 2.5 Encoder-Decoder Structure" width="649" height="819" loading="lazy" decoding="async" />
  <p>Figure 2.5 Encoder-Decoder Structure</p>
</div>

The Transformer consists of an Encoder and a Decoder. Each Encoder (Decoder) in turn consists of 6 Encoder (Decoder) Layers. The input source sequence enters the Encoder for encoding. At the topmost Encoder Layer, the encoding result goes to each of the Decoder Layers. The Decoder decodes the result, and the output target sequence is the final result.

Next, we first introduce the classic structures of traditional neural networks inside the Encoder and Decoder. These structures are the Feed Forward Neural Network (FNN), Layer Normalization (Layer Norm), and the Residual Connection. Then we analyze the internal structures of the Encoder and Decoder.

### 2.2.2 Feed Forward Neural Network

The Feed Forward Neural Network (hereafter abbreviated FNN) is the network structure from the previous section. In this structure, each neuron of a layer connects fully to every neuron of the layers above and below. Each Encoder Layer contains one attention mechanism as above and one feed forward neural network. The feed forward neural network is fairly simple to implement:

```python
class MLP(nn.Module):
    '''Feed Forward Neural Network'''
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        # Define the first linear transformation, from the input dimension to the hidden dimension
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        # Define the second linear transformation, from the hidden dimension to the input dimension
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        # Define a dropout layer to prevent overfitting
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Forward propagation function
        # First, the input x passes through the first linear transformation and the RELU activation function
        # Finally, it passes through the second linear transformation and the dropout layer
        return self.dropout(self.w2(F.relu(self.w1(x))))
    
```

Keep one point in mind about the feed forward neural network of the Transformer. It consists of two linear layers. A RELU activation function sits between them. The feed forward neural network also includes a Dropout layer to prevent overfitting. The Dropout layer is only enabled during training. It is disabled during inference and testing. For this reason, many Transformer architecture diagrams do not draw this layer.

### 2.2.3 Layer Normalization

Layer normalization, that is, Layer Norm, is a classic normalization operation in deep learning. Neural networks have two mainstream normalizations. These are Batch Normalization (Batch Norm) and Layer Normalization (Layer Norm).

The core purpose of normalization is to keep the value range or distribution of the inputs to different layers fairly consistent. In a deep neural network, the input of each layer is the output of the previous layer. The signal propagates through many layers. For the higher layers in the network, parameter changes in all previous neural layers cause the distribution of their inputs to shift considerably. That is to say, as the parameters of the neural network are updated, the output distribution of each layer differs. The differences grow as the network gets deeper. However, the conditional distribution to be predicted is always the same. This mismatch in turn causes prediction error.

Therefore, deep neural networks often need normalization. Normalization makes the input of each layer follow a standard normal distribution. Batch normalization is performed on a mini-batch. A mini-batch is a portion of the samples of a batch. First, the sample mean is computed:

$$
\mu_j = \frac{1}{m}\sum^{m}_{i=1}Z_j^{i}
$$

Here m is the size of the mini-batch. $Z_j^{i}$ is the value of sample i on the j-th dimension.

Then the sample variance is computed:

$$
\sigma^2 = \frac{1}{m}\sum^{m}_{i=1}(Z_j^i - \mu_j)^2
$$

Finally, the process subtracts the mean from each sample value and divides by the standard deviation. This step converts the distribution of this mini-batch of samples into a standard normal distribution:

$$
\widetilde{Z_j} = \frac{Z_j - \mu_j}{\sqrt{\sigma^2 + \epsilon}}
$$

The addition of the extremely small quantity $\epsilon$ here avoids a zero denominator.

However, batch normalization has some flaws. For example:

- When GPU memory is limited and the mini-batch is small, Batch Norm takes the mean and variance from the samples. They cannot reflect the global statistical distribution. This flaw degrades the results.
- RNNs unroll over the time dimension. The same-position distributions of different sentences are most likely different. So Batch Norm's normalization loses its meaning.
- During training, Batch Norm must save the statistics (mean and variance) of each step. At test time, sentences vary in length. The test set contains longer sentences than the training set. So later-position steps have no trained statistics available.
- Batch Norm needs the batch statistics at every step, so it must save and compute them. This work is both time-consuming and laborious.

So Layer Normalization (Layer Norm) emerged. Deep neural networks use it more often, and it works better. Batch Norm computes the mean and variance over all samples at each layer. Layer Norm computes the mean and variance over all layers for each sample. This difference makes each sample's distribution stable. Layer Norm's normalization approach is exactly the same as Batch Norm's. Only the dimension over which the statistics are computed differs.

Based on the normalization formula above, we can implement a Layer Norm layer:

```python
class LayerNorm(nn.Module):
    ''' Layer Norm layer'''
    def __init__(self, features, eps=1e-6):
        super().__init__()
        # Linear matrices for mapping
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
	
    def forward(self, x):
        # Compute the mean and variance over all dimensions of each sample
        mean = x.mean(-1, keepdim=True) # mean: [bsz, max_len, 1]
        std = x.std(-1, keepdim=True) # std: [bsz, max_len, 1]
        # Note that broadcasting also happens over the last dimension here
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
```
The Layer Norm layer above has two linear matrices performing the mapping.

### 2.2.4 Residual Connection

The Transformer's model structure is relatively complex, and the number of layers is deep. To avoid model degradation, the Transformer adopts the idea of residual connections to link every sub-layer. A residual connection means that the input of the next layer is not only the output of the previous layer. It also includes the input of the previous layer. Residual connections allow information from the lowest layer to pass directly to the highest layer. They let the higher layers focus on learning the residuals.

For example, in the Encoder, the input of the first sub-layer first undergoes Layer Normalization (Layer Norm). Then it enters the multi-head self-attention layer, and its output is added to the original input. The second sub-layer works the same way. That is:

$$
x = x + MultiHeadSelfAttention(LayerNorm(x))
$$

$$
output = x + FNN(LayerNorm(x))
$$

The code implementation realizes the residual connection by adding the original value in the forward computation of the layer:

```python
# Attention computation
h = x + self.attention.forward(self.attention_norm(x))
# Pass through the feed forward neural network
out = h + self.feed_forward.forward(self.fnn_norm(h))
```

In the code above, self.attention_norm and self.fnn_norm are both LayerNorm layers. self.attn is the attention layer, and self.feed_forward is the feed forward neural network.

### 2.2.5 Encoder


After you implement the components above, you can build the Encoder of the Transformer. The Encoder consists of N Encoder Layers. Each Encoder Layer includes one attention layer and one feed forward neural network. So implement an Encoder Layer first:

```python
class EncoderLayer(nn.Module):
  '''Encoder layer'''
    def __init__(self, args):
        super().__init__()
        # One Layer contains two LayerNorms, one before the Attention and one before the MLP
        self.attention_norm = LayerNorm(args.n_embd)
        # The Encoder does not need a mask, so pass is_causal=False
        self.attention = MultiHeadAttention(args, is_causal=False)
        self.fnn_norm = LayerNorm(args.n_embd)
        self.feed_forward = MLP(args.dim, args.dim, args.dropout)

    def forward(self, x):
        # Layer Norm
        norm_x = self.attention_norm(x)
        # Self-attention
        h = x + self.attention.forward(norm_x, norm_x, norm_x)
        # Pass through the feed forward neural network
        out = h + self.feed_forward.forward(self.fnn_norm(h))
        return out
```

Then build an Encoder. It is composed of N Encoder Layers, with a Layer Norm added at the end to achieve normalization:

```python
class Encoder(nn.Module):
    '''Encoder block'''
    def __init__(self, args):
        super(Encoder, self).__init__() 
        # One Encoder consists of N Encoder Layers
        self.layers = nn.ModuleList([EncoderLayer(args) for _ in range(args.n_layer)])
        self.norm = LayerNorm(args.n_embd)

    def forward(self, x):
        "Pass through each of the N Encoder Layers in turn"
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
```

The output of the Encoder is the result of encoding the input.

### 2.2.6 Decoder

The same way works here. Build a Decoder Layer first, and then assemble the N Decoder Layers into a Decoder. But the Decoder differs from the Encoder. The Decoder consists of two attention layers and one feed forward neural network. The first attention layer is a masked self-attention layer. It performs the attention computation with a Mask. This layer ensures that each token can only use the attention scores of the tokens before it. The second attention layer is a multi-head attention layer. It uses the output of the first attention layer as the query. It uses the Encoder's output as the key and value to compute the attention scores. Finally, the result passes through the feed forward neural network:

```python
class DecoderLayer(nn.Module):
  '''Decoder layer'''
    def __init__(self, args):
        super().__init__()
        # One Layer contains three LayerNorms, one before Mask Attention, one before Self Attention, and one before the MLP
        self.attention_norm_1 = LayerNorm(args.n_embd)
        # The first part of the Decoder is Mask Attention, so pass is_causal=True
        self.mask_attention = MultiHeadAttention(args, is_causal=True)
        self.attention_norm_2 = LayerNorm(args.n_embd)
        # The second part of the Decoder is Attention similar to the Encoder's, so pass is_causal=False
        self.attention = MultiHeadAttention(args, is_causal=False)
        self.ffn_norm = LayerNorm(args.n_embd)
        # The third part is the MLP
        self.feed_forward = MLP(args.dim, args.dim, args.dropout)

    def forward(self, x, enc_out):
        # Layer Norm
        norm_x = self.attention_norm_1(x)
        # Masked self-attention
        x = x + self.mask_attention.forward(norm_x, norm_x, norm_x)
        # Multi-head attention
        norm_x = self.attention_norm_2(x)
        h = x + self.attention.forward(norm_x, enc_out, enc_out)
        # Pass through the feed forward neural network
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out
```

Then, in the same way, build a Decoder block:

```python
class Decoder(nn.Module):
    '''Decoder'''
    def __init__(self, args):
        super(Decoder, self).__init__() 
        # One Decoder consists of N Decoder Layers
        self.layers = nn.ModuleList([DecoderLayer(args) for _ in range(args.n_layer)])
        self.norm = LayerNorm(args.n_embd)

    def forward(self, x, enc_out):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, enc_out)
        return self.norm(x)
```

The Encoder and Decoder above complete the core part of the Transformer. Next, concatenate the Encoder and Decoder and add the Embedding layers. The result is the complete Transformer model.

## 2.3 Building a Transformer

The previous two sections analyzed the Attention mechanism and the core of the Transformer in depth. This core is the Encoder and Decoder structures. Next, you can build a complete Transformer model from the components implemented in the previous section.

### 2.3.1 The Embedding Layer

Chapter 1 discussed that NLP tasks often need to convert the natural language input into vectors that a machine can process. In deep learning, the Embedding layer undertakes this task.

The Embedding layer is essentially a lookup table of embedding vectors for a fixed-size vocabulary. Before the input goes into the neural network, we usually pass the natural language input through a tokenizer first. The tokenizer splits the natural language input into tokens and converts them into fixed indices. For example, set the vocabulary size to 4 and use “我喜欢你” (I like you) as the input. The tokenizer can then convert the input into:

```
input: 我 (I)
output: 0

input: 喜欢 (like)
output: 1

input：你 (you)
output: 2
```

In practice, the tokenizer's work is more complicated than this example. For example, many different tokenization methods exist. These methods split into words, split into subwords, or split into characters. The vocabulary size often reaches tens or even hundreds of thousands. This section does not elaborate on the tokenizer's details. Later, we will introduce in detail how the tokenizer of a large model operates and is trained.

So the Embedding layer's input is usually a matrix of shape (batch_size, seq_len, 1). The first dimension is the number of samples processed in one batch. The second dimension is the length of the natural language sequence. The third dimension is the index value to which the tokenizer converted each token. For the input above, the Embedding layer's input is:

```
[[[0],[1],[2]]]
```

Its batch_size is 1. Its seq_len is 3, and the converted indices are as above.

Internally, the Embedding is a trainable (Vocab_size, embedding_dim) weight matrix. Every entry in the vocabulary corresponds to one row, which is a vector of dimension embedding_dim. For an input value, the layer looks up the corresponding word vector. Then it concatenates these vectors into a (batch_size, seq_len, embedding_dim) matrix as the output.

The implementation above is not complicated. We can use the Embedding layer from torch directly:

```python
self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
```

### 2.3.2 Positional Encoding

The attention mechanism achieves good parallel computation. At the same time, its attention computation also causes the loss of relative positions in the sequence. RNNs and LSTMs process the input sequence one step at a time along the sentence's own order. So the order of the input sequence carries extremely important information. This order also fits well with the inherent nature of natural language.

However, the analysis of the attention mechanism above shows one property. In the attention computation, all other positions are equal to a given token in the sequence. That is, “我喜欢你” (I like you) and “你喜欢我” (You like me) look completely identical to the attention mechanism. But this equality is a huge problem of the attention mechanism. So the Transformer adopts a positional encoding mechanism. This mechanism uses sequence order information and preserves the relative position information in the sequence. Many models later adopted it.

Positional encoding means encoding each token according to its relative position in the sequence. Then the mechanism adds the positional encoding to the word vector encoding. There are many ways to perform positional encoding. The Transformer uses sine and cosine functions for positional encoding (absolute positional encoding, Sinusoidal). The encoding is as follows:

$$
PE(pos, 2i) = sin(pos/10000^{2i/d_{model}})\\
PE(pos, 2i+1) = cos(pos/10000^{2i/d_{model}})
$$

In the formula above, pos is the token's position in the sentence. The values 2i and 2i+1 indicate whether the positional encoding vector's dimension index is odd or even. The formula above shows that the Transformer uses different functions for encoding in odd and even dimensions.

A simple example illustrates the positional encoding computation. Suppose the input is a sentence of length 4, "I like to code". We can obtain the following word vector matrix $\rm x$, where each row represents one word vector. $\rm x_0=[0.1,0.2,0.3,0.4]$ corresponds to the word vector of "I", whose pos is 0. The next rows work the same way. The second row represents the word vector of "like", whose pos is 1:

$$
\rm x = \begin{bmatrix} 
0.1 & 0.2 & 0.3 & 0.4 \\   
0.2 & 0.3 & 0.4 & 0.5 \\    
0.3 & 0.4 & 0.5 & 0.6 \\    
0.4 & 0.5 & 0.6 & 0.7
\end{bmatrix}
$$

So the word vectors after positional encoding are:

$$
\rm x_{PE} = \begin{bmatrix} 
0.1 & 0.2 & 0.3 & 0.4 \\     
0.2 & 0.3 & 0.4 & 0.5 \\    
0.3 & 0.4 & 0.5 & 0.6 \\     
0.4 & 0.5 & 0.6 & 0.7 
\end{bmatrix} + \begin{bmatrix}
\sin(\frac{0}{10000^0}) & \cos(\frac{0}{10000^0}) & \sin(\frac{0}{10000^{2/4}}) & \cos(\frac{0}{10000^{2/4}}) \\ 
\sin(\frac{1}{10000^0}) & \cos(\frac{1}{10000^0}) & \sin(\frac{1}{10000^{2/4}}) & \cos(\frac{1}{10000^{2/4}}) \\ 
\sin(\frac{2}{10000^0}) & \cos(\frac{2}{10000^0}) & \sin(\frac{2}{10000^{2/4}}) & \cos(\frac{2}{10000^{2/4}}) \\ 
\sin(\frac{3}{10000^0}) & \cos(\frac{3}{10000^0}) & \sin(\frac{3}{10000^{2/4}}) & \cos(\frac{3}{10000^{2/4}}) 
\end{bmatrix} = \begin{bmatrix} 
0.1 & 1.2 & 0.3 & 1.4 \\ 
1.041 & 0.84 & 0.41 & 1.49 \\ 
1.209 & -0.016 & 0.52 & 1.59 \\ 
0.541 & -0.489 & 0.895 & 1.655 
\end{bmatrix}
$$

The following code obtains the positional encoding of the example above:
```python
import numpy as np
import matplotlib.pyplot as plt
def PositionEncoding(seq_len, d_model, n=10000):
    P = np.zeros((seq_len, d_model))
    for k in range(seq_len):
        for i in np.arange(int(d_model/2)):
            denominator = np.power(n, 2*i/d_model)
            P[k, 2*i] = np.sin(k/denominator)
            P[k, 2*i+1] = np.cos(k/denominator)
    return P

P = PositionEncoding(seq_len=4, d_model=4, n=100)
print(P)
```

```python
[[ 0.          1.          0.          1.        ]
 [ 0.84147098  0.54030231  0.09983342  0.99500417]
 [ 0.90929743 -0.41614684  0.19866933  0.98006658]
 [ 0.14112001 -0.9899925   0.29552021  0.95533649]]
```

Such positional encoding has two main benefits:

1. It allows the PE to adapt to sentences longer than any in the training set. Suppose the longest sentence in the training set has 20 words, and suddenly a sentence of length 21 arrives. The formula-based computation method can compute the Embedding of the 21st position.
2. It allows the model to compute relative positions easily. For a fixed-length offset k, the model can compute PE(pos+k) from PE(pos). Sin(A+B) = Sin(A)Cos(B) + Cos(A)Sin(B) and Cos(A+B) = Cos(A)Cos(B) - Sin(A)Sin(B) give the reason.

We can also prove the superiority of this encoding method through a rigorous mathematical derivation. The original Transformer Embedding can be expressed as:

$$
\begin{equation}f(\cdots,\boldsymbol{x}_m,\cdots,\boldsymbol{x}_n,\cdots)=f(\cdots,\boldsymbol{x}_n,\cdots,\boldsymbol{x}_m,\cdots)\end{equation}
$$

Clearly, such a function is not asymmetric. This property means it cannot represent relative position information. We want to obtain the following kind of encoding:

$$
\begin{equation}\tilde{f}(\cdots,\boldsymbol{x}_m,\cdots,\boldsymbol{x}_n,\cdots)=f(\cdots,\boldsymbol{x}_m + \boldsymbol{p}_m,\cdots,\boldsymbol{x}_n + \boldsymbol{p}_n,\cdots)\end{equation}
$$

Here, the added $p_m$ and $p_n$ are the positional encodings. Next, Taylor-expand $f(...,x_m+p_m,...,x_n+p_n)$ at the two positions m and n:

$$
\begin{equation}\tilde{f}\approx f + \boldsymbol{p}_m^{\top} \frac{\partial f}{\partial \boldsymbol{x}_m} + \boldsymbol{p}_n^{\top} \frac{\partial f}{\partial \boldsymbol{x}_n} + \frac{1}{2}\boldsymbol{p}_m^{\top} \frac{\partial^2 f}{\partial \boldsymbol{x}_m^2}\boldsymbol{p}_m + \frac{1}{2}\boldsymbol{p}_n^{\top} \frac{\partial^2 f}{\partial \boldsymbol{x}_n^2}\boldsymbol{p}_n + \underbrace{\boldsymbol{p}_m^{\top} \frac{\partial^2 f}{\partial \boldsymbol{x}_m \partial \boldsymbol{x}_n}\boldsymbol{p}_n}_{\boldsymbol{p}_m^{\top} \boldsymbol{\mathcal{H}} \boldsymbol{p}_n}\end{equation}
$$

The 1st term is independent of position. Terms 2–5 depend on only a single position. The 6th term involves both positions, where f is partially differentiated with respect to m and n respectively. So we hope the 6th term ($p_m^THp_n$) expresses the relative position information. We seek a function g such that:

$$
p_m^THp_n = g(m-n)
$$

Suppose $H$ is an identity matrix. Then:

$$
p_m^THp_n = p_m^Tp_n = \langle\boldsymbol{p}_m, \boldsymbol{p}_n\rangle = g(m-n)
$$

Treat the vector [x,y] as the complex number x+yi. Then construct equations based on the rules of complex number operations:

$$
\begin{equation}\langle\boldsymbol{p}_m, \boldsymbol{p}_n\rangle = \text{Re}[\boldsymbol{p}_m \boldsymbol{p}_n^*]\end{equation}
$$

Further suppose a complex number $q_{m-n}$ exists such that:

$$
\begin{equation}\boldsymbol{p}_m \boldsymbol{p}_n^* = \boldsymbol{q}_{m-n}\end{equation}
$$

Solving this equation with the exponential form of complex numbers yields the solution of the positional encoding in the two-dimensional case:

$$
\begin{equation}\boldsymbol{p}_m = e^{\text{i}m\theta}\quad\Leftrightarrow\quad \boldsymbol{p}_m=\begin{pmatrix}\cos m\theta \\ \sin m\theta\end{pmatrix}\end{equation}
$$

The inner product satisfies linear superposition. So the positional encoding of a higher-dimensional even dimension can be expressed as a combination of multiple two-dimensional positional encodings:

$$
\begin{equation}\boldsymbol{p}_m = \begin{pmatrix}e^{\text{i}m\theta_0} \\ e^{\text{i}m\theta_1} \\ \vdots \\ e^{\text{i}m\theta_{d/2-1}}\end{pmatrix}\quad\Leftrightarrow\quad \boldsymbol{p}_m=\begin{pmatrix}\cos m\theta_0 \\ \sin m\theta_0 \\ \cos m\theta_1 \\ \sin m\theta_1 \\ \vdots \\ \cos m\theta_{d/2-1} \\ \sin m\theta_{d/2-1}  \end{pmatrix}\end{equation}
$$

Then take $\theta_i = 10000^{-2i/d}$. This form ensures that as |m−n| grows, ⟨pm,pn⟩ exhibits a trend toward zero. You can prove this trend by integrating over the positional encodings. The base of 10000 is an experimental result. This choice yields the encoding method above.

Here $H$ is not an identity matrix. The correlation between any two of the d-dimensional vectors formed by the Embedding layer of the model is relatively small. The vectors satisfy a certain degree of decoupling, so we can regard the matrix as a diagonal matrix. Then use the encoding above:

$$
\begin{equation}\boldsymbol{p}_m^{\top} \boldsymbol{\mathcal{H}} \boldsymbol{p}_n=\sum_{i=1}^{d/2} \boldsymbol{\mathcal{H}}_{2i,2i} \cos m\theta_i \cos n\theta_i + \boldsymbol{\mathcal{H}}_{2i+1,2i+1} \sin m\theta_i \sin n\theta_i\end{equation}
$$

Through product-to-sum identities:

$$
\begin{equation}\sum_{i=1}^{d/2} \frac{1}{2}\left(\boldsymbol{\mathcal{H}}_{2i,2i} + \boldsymbol{\mathcal{H}}_{2i+1,2i+1}\right) \cos (m-n)\theta_i + \frac{1}{2}\left(\boldsymbol{\mathcal{H}}_{2i,2i} - \boldsymbol{\mathcal{H}}_{2i+1,2i+1}\right) \cos (m+n)\theta_i \end{equation}
$$

This result shows that this encoding can still represent relative position.

Figure 2.6 shows the encoding results above:

<div align="center">
  <img src="/images/2-figures/3-0.webp" alt="Figure 2.6 Encoding Results" width="1356" height="626" loading="lazy" decoding="async" />
  <p>Figure 2.6 Encoding Results</p>
</div>


Based on the principle above, we implement a positional encoding layer:

```python

class PositionalEncoding(nn.Module):
    '''Positional encoding module'''

    def __init__(self, args):
        super(PositionalEncoding, self).__init__()
        # Dropout layer
        # self.dropout = nn.Dropout(p=args.dropout)

        # block size is the maximum length of the sequence
        pe = torch.zeros(args.block_size, args.n_embd)
        position = torch.arange(0, args.block_size).unsqueeze(1)
        # Compute theta
        div_term = torch.exp(
            torch.arange(0, args.n_embd, 2) * -(math.log(10000.0) / args.n_embd)
        )
        # Compute the sin and cos results respectively
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # Add the positional encoding onto the Embedding result
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return x
```

### 2.3.3 A Complete Transformer

Concatenate all of the components above according to the Transformer structure in the figure below. The result is a complete Transformer model, as shown in Figure 2.7:

<div align="center">
  <img src="/images/2-figures/3-1.webp" alt="Figure 2.7 Transformer Model Structure" width="636" height="847" loading="lazy" decoding="async" />
  <p>Figure 2.7 Transformer Model Structure</p>
</div>

But keep one property of the figure above in mind. The figure above is the illustration from the original paper "Attention is all you need". In this illustration, the LayerNorm layer sits after the Attention layer, which is the "Post-Norm" structure. In the released source code, the LayerNorm layer sits before the Attention layer, which is the "Pre Norm" structure. Current LLMs generally adopt the "Pre-Norm" structure, which makes the loss more stable. So this chapter adopts the "Pre-Norm" structure in its implementation.

As the figure shows, the output first passes through the tokenizer. Then it passes through the Embedding layer and the Positional Embedding layer for encoding. Then it enters the N Encoders and N Decoders from the previous section (in the original Transformer model, N is set to 6). Finally it passes through a linear layer and a Softmax layer to obtain the final output.

Based on the components implemented previously, we implement the complete Transformer model:

```python
class Transformer(nn.Module):
   '''Overall model'''
    def __init__(self, args):
        super().__init__()
        # The vocabulary size and block size must be provided
        assert args.vocab_size is not None
        assert args.block_size is not None
        self.args = args
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(args.vocab_size, args.n_embd),
            wpe = PositionalEncoding(args),
            drop = nn.Dropout(args.dropout),
            encoder = Encoder(args),
            decoder = Decoder(args),
        ))
        # The final linear layer: the input is n_embd, and the output is the vocabulary size
        self.lm_head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

        # Initialize all weights
        self.apply(self._init_weights)

        # Print the total number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    '''Count all parameters'''
    def get_num_params(self, non_embedding=False):
        # non_embedding: whether to count the embedding parameters
        n_params = sum(p.numel() for p in self.parameters())
        # If the embedding parameters are not counted, subtract them
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()
        return n_params

    '''Initialize weights'''
    def _init_weights(self, module):
        # Linear layers and the Embedding layer are initialized with a normal distribution
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    '''Forward computation function'''
    def forward(self, idx, targets=None):
        # The input is idx with dimension (batch size, sequence length, 1); targets is the target sequence, used to compute the loss
        device = idx.device
        b, t = idx.size()
        assert t <= self.args.block_size, f"Cannot compute this sequence: its length is {t}, while the maximum sequence length is only {self.args.block_size}"

        # Pass through self.transformer
        # First pass the input idx through the Embedding layer, obtaining the dimension (batch size, sequence length, n_embd)
        print("idx",idx.size())
        # Pass through the Embedding layer
        tok_emb = self.transformer.wte(idx)
        print("tok_emb",tok_emb.size())
        # Then pass through the positional encoding
        pos_emb = self.transformer.wpe(tok_emb) 
        # Then apply Dropout
        x = self.transformer.drop(pos_emb)
        # Then pass through the Encoder
        print("x after wpe:",x.size())
        enc_out = self.transformer.encoder(x)
        print("enc_out:",enc_out.size())
        # Then pass through the Decoder
        x = self.transformer.decoder(x, enc_out)
        print("x after decoder:",x.size())

        if targets is not None:
            # Training phase: if we are given targets, compute the loss
            # First pass through the final Linear layer, obtaining the dimension (batch size, sequence length, vocab size)
            logits = self.lm_head(x)
            # Then compute the cross entropy against targets
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Inference phase: we only need the logits, and the loss is None
            # Taking -1 means only the last element of the sequence is taken as the output
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss
```

Keep one point in mind about the code above. It builds the entire Transformer structure. It also implements three functions:

- get_num_params: used to count the number of parameters of the model
- _init_weights: used to randomly initialize all of the parameters of the model
- forward: the forward computation function

In addition, the forward computation function uses the cross-entropy function of PyTorch to compute the loss. Readers can consult the official documentation of PyTorch for other loss functions. This section does not elaborate on them.

After the steps above, we can "hand-build" a complete, computable Transformer model from scratch. This book focuses mainly on LLMs. So this chapter does not go into detail on how to train the Transformer model. Later chapters will "hand-build" a LLaMA model from scratch in the same way. They will walk you step by step through training your own Tiny LLaMA.

The sources for this chapter are listed below.

[1] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin. (2023). *Attention Is All You Need.* arXiv preprint arXiv:1706.03762.

[2] Jay Mody's article "An Intuition for Attention". Source: https://jaykmody.com/blog/attention-intuition/
