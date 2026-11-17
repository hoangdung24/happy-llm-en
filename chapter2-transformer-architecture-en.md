# Chapter 2: Transformer Architecture

## 2.1 The Attention Mechanism

### 2.1.1 What is the Attention Mechanism

As NLP advances from statistical machine learning toward deep learning, text representation—the core problem of NLP—has likewise gradually advanced from statistical learning to deep learning. As we introduced in Chapter 1, text representation evolved from the early vector space models and language models computed via statistical learning, through the single-layer neural network of Word2Vec, into the era of learning text representations with neural networks. However, neural networks, which originated from Computer Vision (CV), have three core architectures:

- Feedforward Neural Network (FNN), i.e., data flows unidirectionally from the input layer to the output layer with no recurrent structure; layers pass information to each other through full connections or specific schemes. The Multi-Layer Perceptron (MLP) is the most common form, where the neurons of each layer are fully connected to every neuron of the layers above and below, as shown in Figure 2.1:

<div align="center">
  <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/2-figures/1-0.png" alt="Figure caption" width="90%"/>
  <p>Figure 2.1 Fully Connected Neural Network</p>
</div>

- Convolutional Neural Network (CNN), i.e., convolutional layers—whose number of trainable parameters is far smaller than that of a fully connected neural network—are used for feature extraction and learning, as shown in Figure 2.2:

<div align="center">
  <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/2-figures/1-1.png" alt="Figure caption" width="90%"/>
  <p>Figure 2.2 Convolutional Neural Network</p>
</div>

- Recurrent Neural Network (RNN), a network that can use historical information as input and contains loops and self-repetition, as shown in Figure 2.3:

<div align="center">
  <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/2-figures/1-2.png" alt="Figure caption" width="90%"/>
  <p>Figure 2.3 Recurrent Neural Network</p>
</div>

Since the text processed by NLP tasks is usually sequential, RNNs—specialized for sequential and time-series data—often achieve the best results on NLP tasks. In fact, before the attention mechanism burst onto the scene, RNNs and their derivative architecture LSTM were the undisputed rulers of the NLP field. For example, ELMo—the text representation model that pioneered the idea of pre-training, which we discussed in Chapter 1—uses a bidirectional LSTM as its network architecture.

However, although RNNs and LSTMs have the advantages of capturing temporal information and being well suited to sequence generation, they have two hard-to-fix flaws:

1. The sequential computation pattern models temporal information well, but it limits the parallel computing capability of computers. Since sequences must be input and computed in order, the parallel computing capability of Graphics Processing Units (GPUs) is greatly restricted; as a result, models built on RNN architectures, although not particularly large in parameter count, incur high computational time costs;

2. RNNs struggle to capture correlations over long sequences. In the RNN architecture, the farther apart two inputs are, the harder their relationship is to capture; meanwhile, an RNN must read the entire sequence into memory and compute it in order, which also limits sequence length. Although LSTMs optimize this to some extent through gating mechanisms, RNNs remain unsatisfactory at capturing long-distance correlations.

To address these problems, Vaswani et al. drew on the attention mechanism—proposed in the CV field and often integrated into RNNs (note: although the attention mechanism flourished in NLP, it was indeed first proposed in the CV field)—and innovatively built a neural network composed entirely of attention mechanisms: the Transformer, the ancestor and core architecture of the Large Language Model (LLM), which instantly made the attention mechanism one of the most core architectures in deep learning.

So what exactly is the attention mechanism?

The attention mechanism originated in the computer vision field. Its core idea is that when we look at an image, we often do not need to see all of its content—we can simply focus our attention on the key parts. In the field of natural language processing, we can likewise achieve more efficient, higher-quality computation by focusing attention on one or a few tokens.

The attention mechanism has three core variables: **Query** (the query value), **Key** (the key value), and **Value** (the true value). We can understand what each variable represents through an example. Suppose we have a news report and want to find the time of the event. Then our Query could be a vector resembling "time" or "date" (text is used here for ease of understanding, but it is actually a dense vector), while the Key and the Value would be the entire text. By computing on the Query and the Key, we obtain a weight; this weight actually reflects, starting from the Query, the relative magnitude of attention that should be distributed over each token of the text. By computing on the weight and the Value, the final result is what we obtain by computing the attention of the entire text starting from the Query.

​Specifically, the attention mechanism works by computing the correlation between **Query** and **Key** to form a weighted sum of the true values, thereby fitting the correlation between each word in the sequence and every other word.

### 2.1.2 A Deeper Understanding of the Attention Mechanism

We just said that the attention mechanism has three core variables: the query value Query, the key value Key, and the true value Value. Next, using a dictionary as an example, we analyze step by step how the attention computation formula is derived, helping readers deeply understand the attention mechanism. First, consider the following dictionary:

```json
{
    "apple":10,
    "banana":5,
    "chair":2
}
```

Here, the dictionary's keys are the Keys of the attention mechanism, and the dictionary's values are the Values. A dictionary supports exact string matching: for example, if the value we want to look up—i.e., the query value Query—is "apple", we can directly match the Query against the Key to obtain the corresponding Value.

But what if the Query we want to match is a concept containing multiple Keys? For example, we want to look up "fruit". In that case, we should match both apple and banana, but not chair. Therefore, we usually combine the Values corresponding to the Keys to obtain the final Value.

For example, when our Query is "fruit", we can assign the following weights to the three Keys:

```json
{
    "apple":0.6,
    "banana":0.4,
    "chair":0
}
```

Then the value we finally retrieve should be:

$$
value = 0.6 * 10 + 0.4 * 5 + 0 * 2 = 8
$$

The different weights assigned to different Keys are what we call attention scores—that is, how much attention we should give to each Key in order to answer the Query. But how do we compute the corresponding attention scores for each Query? Intuitively, the more relevant a Key is to the Query, the larger the attention weight it should receive. But how can we find a reasonable method that computes correct attention scores?

In Chapter 1, we mentioned the concept of word vectors. Through proper training and fitting, word vectors can represent semantic information, so that semantically similar words are closer together in vector space, while semantically distant words are farther apart. We usually use Euclidean distance to measure the similarity of word vectors, but we can equally measure it with the dot product:

$$
v·w = \sum_{i}v_iw_i
$$

By the definition of word vectors, the dot product of the word vectors of two semantically similar words should be greater than 0, while the dot product of semantically dissimilar word vectors should be less than 0.

Thus, we can use the dot product to compute the similarity between words. Suppose our Query is "fruit" with corresponding word vector $q$; our Keys correspond to the word vectors $k = [v_{apple} v_{banana} v_{chair}]$. Then we can compute the similarity of the Query with each key:

$$
x = qK^T
$$

Here, K is the matrix formed by stacking the word vectors of all the Keys. By the definition of matrix multiplication, x is the dot product of q with each k. The x we now obtain reflects the similarity between the Query and each Key; we then pass it through a Softmax layer to convert it into weights summing to 1:

$$
\text{softmax}(x)_i = \frac{e^{xi}}{\sum_{j}e^{x_j}}
$$

In this way, the resulting vector reflects the similarity between the Query and each Key, and at the same time the weights sum to 1—that is, our attention scores. Finally, we multiply the obtained attention scores by the value vectors correspondingly. From the process above, we can obtain the basic formula of attention computation:

$$
attention(Q,K,V) = softmax(qK^T)v
$$

However, the value is still a scalar at this point, and in this pass we queried only one Query. We can convert the value into a vector of dimension $d_v$, and query multiple Queries at once by stacking the word vectors of the multiple Queries into a matrix Q, yielding the formula:

$$
attention(Q,K,V) = softmax(QK^T)V
$$

We are now one final step away from the standard attention formula. In the previous formula, if Q and K have a relatively large dimension $d_k$, the softmax scaling is very easily affected, making the differences between values large and thus affecting gradient stability. Therefore, we scale the product of Q and K:

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

Note that in the code above, we assume the inputs q, k, v are word vector matrices that have already been transformed, i.e., the Q, K, V in the formula. With just these few lines of code, we can implement the core attention computation.

### 2.1.4 Self-Attention

From the analysis above, we can see that the essence of the attention mechanism is to sequentially compute similarity between the elements of two sequences, finding the relevance of each element of one sequence to each element of the other, and then weighting based on that relevance—that is, allocating attention. These two sequences are the sources of Q, K, V in our computation.

In practical applications, however, we often only need to compute the attention result between the Query and the Key; an additional true value Value rarely exists. That is, we actually only need to fit two text sequences. In the classic attention mechanism, Q often comes from one sequence while K and V come from another, both computed through parameter matrices, so that the relationship between the two sequences can be fitted. For example, in the Transformer's Decoder structure, Q comes from the Decoder's input while K and V come from the Encoder's output, thereby fitting the relationship between the encoded information and the historical information, which facilitates combining both kinds of information to make predictions about the future.

​But the Transformer's Encoder structure uses a variant of the attention mechanism—self-attention. Self-attention computes the attention distribution of each element of the sequence itself over the other elements; that is, in the computation, Q, K, V are all computed from the same input through different parameter matrices. In the Encoder, Q, K, V are obtained by multiplying the input with the parameter matrices $W_q$, $W_k$, $W_v$ respectively, thereby fitting the relationship of each token in the input sentence to all other tokens.

Through the self-attention mechanism, we can find the magnitude of the correlation between each token in a text and all other tokens, thereby modeling the dependencies within the text. In code, the self-attention mechanism is actually implemented by passing the same parameter as the input to Q, K, V:

```python
# attention is the attention computation function defined above
attention(x, x, x)
```

### 2.1.5 Masked Self-Attention

Masked self-attention, i.e., Mask Self-Attention, refers to the self-attention mechanism that uses an attention mask. The mask's role is to hide tokens at certain positions; during learning, the model ignores the masked tokens.

The core motivation for using an attention mask is to let the model use only historical information for prediction, without seeing future information. Transformer models using the attention mechanism also learn through a language-modeling task similar to n-gram: given a text sequence, the model continuously predicts the next token based on the previous tokens until the entire text sequence is completed.

For example, if the text sequence to be learned is 【BOS】I like you【EOS】, then the model predicts and learns in the following order:

    Step 1: input 【BOS】, output I
    Step 2: input 【BOS】I, output like
    Step 3: input 【BOS】I like, output you
    Step 4: input 【BOS】I like you, output 【EOS】

In theory, as long as the training corpus is large enough, through the above process the model can learn to model any kind of text sequence, i.e., it can complete any text.

However, we can see that the above process is serial: Step 1 must be completed before Step 2 can be done, and so on until the entire sequence is completed. We said at the very beginning that one of the Transformer's core advantages over the RNN is that it can compute in parallel, with higher computational efficiency. If, for each training sample, the model has to complete the above process serially in order to finish learning, then clearly parallel computation is not achieved and computational efficiency is low.

To solve this problem, the Transformer proposed the masked self-attention method. Masked self-attention generates a mask to hide future information. For example, the text sequence to be learned is still 【BOS】I like you【EOS】, and the attention mask we use is 【MASK】; then the model's input is:

    <BOS> 【MASK】【MASK】【MASK】【MASK】
    <BOS>    I   【MASK】 【MASK】【MASK】
    <BOS>    I     like  【MASK】【MASK】
    <BOS>    I     like    you  【MASK】
    <BOS>    I     like    you   </EOS>

In each row of input, the model still only sees the preceding tokens and predicts the next token. But note that these inputs are no longer a serial process—they can be fed into the model in parallel all at once; the model only needs each sample to predict the next token based on the unmasked tokens, thereby realizing a parallel language model.

Observing the mask above, we can see that it is actually an upper-triangular matrix with the same length as the text sequence. We can simply create an upper-triangular matrix of the same length as the input to serve as the attention mask, then use the mask to mask out parts of the input. That is, when the input dimension is (batch_size, seq_len, hidden_size), our Mask matrix generally has dimension (1, seq_len, seq_len) (broadcasting enables the computation for different samples in the same batch).

In the concrete implementation, we generate the Mask matrix with the following code:

```python
# Create an upper-triangular matrix to hide future information.
# First create a 1 * seq_len * seq_len matrix with the full function
mask = torch.full((1, args.max_seq_len, args.max_seq_len), float("-inf"))
# The triu function creates an upper-triangular matrix
mask = torch.triu(mask, diagonal=1)
```

The generated Mask matrix is an upper-triangular matrix: the elements in the upper-triangular positions are all -inf, and the elements elsewhere are set to 0.

During attention computation, we add the computed attention scores to this mask and then perform the Softmax operation:

```python
# Here scores are the computed attention scores, and mask is the mask matrix generated above
scores = scores + mask[:, :seqlen, :seqlen]
scores = F.softmax(scores.float(), dim=-1).type_as(xq)
```

After the addition, the attention scores in the upper-triangular region (i.e., the positions corresponding to the tokens that should be masked) all become `-inf`, while the scores in the lower-triangular region are unchanged. After the Softmax operation, the `-inf` values are set to 0, so the attention scores computed for the upper-triangular region are ignored, thereby realizing attention masking.

### 2.1.6 Multi-Head Attention

The attention mechanism can achieve parallelization and fit long-term dependencies, but a single attention computation can only fit one kind of correlation; a single attention mechanism can hardly comprehensively fit the correlations in a sentence sequence. Therefore, the Transformer uses the Multi-Head Attention mechanism: it performs multiple attention computations on the same corpus simultaneously, each attention computation fitting a different relation, and concatenates the multiple final results as the final output, thereby fitting the linguistic information more comprehensively and deeply.

In the original paper, the authors also confirmed experimentally that, in multi-head attention computation, each different attention head can fit different information in the sentence, as shown in Figure 2.4:

<div align="center">
  <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/2-figures/1-3.jpeg" alt="Figure caption" width="90%"/>
  <p>Figure 2.4 Multi-Head Attention Mechanism</p>
</div>

​The upper and lower parts show the results of two attention heads computing self-attention on the same sentence sequence. As we can see, different attention heads can fit different levels of correlated information. Computing with multiple attention heads simultaneously fits the sentence's relations more comprehensively.

In fact, the so-called multi-head attention mechanism is simply performing multiple groups of self-attention processing on the original input sequence; then the self-attention results of each group are concatenated and processed through a linear layer to obtain the final output. Formally, we can express it as:

$$
\mathrm{MultiHead}(Q, K, V) = \mathrm{Concat}(\mathrm{head_1}, ...,
\mathrm{head_h})W^O    \\
    \text{where}~\mathrm{head_i} = \mathrm{Attention}(QW^Q_i, KW^K_i, VW^V_i)
$$

Its most straightforward code implementation is not complicated: n heads means n groups of 3 parameter matrices; each group performs the same attention computation, but because the parameter matrices are different, different attention results are achieved through backpropagation; then the n results are concatenated and output.

However, the above implementation has high time and space complexity. We can cleverly implement parallel multi-head computation through matrix operations; the core logic is to use three combined matrices to replace the combination of n parameter matrices—that is, inner product then concatenation is actually equivalent to concatenation then inner product. For the concrete implementation, refer to the following code:

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

In the previous section, we introduced in detail the core of the Transformer—the attention mechanism. In the paper "Attention is All You Need", the authors built the Transformer model using only the attention mechanism while abandoning the traditional RNN and CNN architectures, bringing a great transformation to the NLP field. In the Transformer, the attention mechanism is used by its two core components—the Encoder (encoder) and the Decoder (decoder). In fact, the pre-trained language models that later derived from the Transformer architecture are basically built by improving the Encoder-Decoder part to construct new model architectures, for example BERT, which uses only the Encoder, GPT, which uses only the Decoder, and so on.

In this section, based on the attention mechanism introduced in the previous section, we start from the Seq2Seq task that the Transformer targets and analyze the Transformer's Encoder-Decoder structure.

### 2.2.1 Seq2Seq Models

Seq2Seq, i.e., sequence-to-sequence, is a classic NLP task. Specifically, the model inputs a natural language sequence $input = (x_1, x_2, x_3...x_n)$ and outputs a natural language sequence of possibly different length $output = (y_1, y_2, y_3...y_m)$. In fact, Seq2Seq is the most classic NLP task; almost all NLP tasks can be viewed as Seq2Seq tasks. For example, a text classification task can be viewed as outputting a target sequence of length 1 (i.e., $m$ = 1 in the formula above); a part-of-speech tagging task can be viewed as outputting a target sequence of the same length as the input sequence (i.e., $m$ = $n$ in the formula above).

The machine translation task is a classic Seq2Seq task: for example, our input might be “今天天气真好” (The weather is really nice today) and the output "Today is a good day.". The Transformer is a classic Seq2Seq model: the model's input is one text sequence, and its output is another text sequence. In fact, the Transformer was originally applied precisely to the machine translation task.

For Seq2Seq tasks, the general approach is to encode the natural language sequence and then decode it. Encoding means encoding the input natural language sequence through hidden layers into a vector (or matrix) that represents semantics, which can be simply understood as a more complex word-vector representation. Decoding means passing the vector or matrix obtained by encoding the input natural language sequence through hidden layers, and then decoding it into the corresponding natural language target sequence. Through encoding and then decoding, the Seq2Seq task can be accomplished.

The Encoder in the Transformer performs the encoding process described above; the Decoder performs the decoding process. The Transformer structure is shown in Figure 2.5:

<div align="center">
  <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/2-figures/2-0.jpg" alt="Figure caption" width="90%"/>
  <p>Figure 2.5 Encoder-Decoder Structure</p>
</div>

The Transformer consists of an Encoder and a Decoder, and each Encoder (Decoder) in turn consists of 6 Encoder (Decoder) Layers. The input source sequence enters the Encoder for encoding; at the topmost Encoder Layer, the encoding result is output to each of the Decoder Layers, and after being decoded by the Decoder, the output target sequence is obtained.

Next, we will first introduce the classic structures of traditional neural networks inside the Encoder and Decoder—the Feed Forward Neural Network (FNN), Layer Normalization (Layer Norm), and the Residual Connection—and then further analyze the internal structures of the Encoder and Decoder.

### 2.2.2 Feed Forward Neural Network

The Feed Forward Neural Network (hereafter abbreviated FNN), as mentioned in the previous section, is the network structure in which the neurons of each layer are fully connected to every neuron of the layers above and below. Each Encoder Layer contains one attention mechanism as described above and one feed forward neural network. The feed forward neural network is fairly simple to implement:

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

Note that the Transformer's feed forward neural network consists of two linear layers with a RELU activation function in between, and the feed forward neural network also includes a Dropout layer to prevent overfitting. The Dropout layer is only enabled during training and disabled during inference/testing, so many Transformer architecture diagrams do not draw this layer.

### 2.2.3 Layer Normalization

Layer normalization, i.e., Layer Norm, is a classic normalization operation in deep learning. The two mainstream normalizations in neural networks are Batch Normalization (Batch Norm) and Layer Normalization (Layer Norm).

The core purpose of normalization is to keep the value range or distribution of the inputs to different layers fairly consistent. Since the input of each layer in a deep neural network is the output of the previous layer, over the propagation through many layers, for the higher layers in the network, parameter changes in all previous neural layers cause the distribution of their inputs to shift considerably. That is to say, as the neural network's parameters are updated, the output distribution of each layer differs, and the differences grow as the network gets deeper. However, the conditional distribution to be predicted is always the same, which in turn causes prediction error.

Therefore, in deep neural networks, normalization is often needed, so that the input of each layer is normalized into a standard normal distribution. Batch normalization is performed on a mini-batch, which is equivalent to splitting out a portion of the samples of a batch. First, the sample mean is computed:

$$
\mu_j = \frac{1}{m}\sum^{m}_{i=1}Z_j^{i}
$$

where $Z_j^{i}$ is the value of sample i on the j-th dimension, and m is the size of the mini-batch.

Then the sample variance is computed:

$$
\sigma^2 = \frac{1}{m}\sum^{m}_{i=1}(Z_j^i - \mu_j)^2
$$

Finally, each sample's value is subtracted by the mean and divided by the standard deviation, converting the distribution of this mini-batch of samples into a standard normal distribution:

$$
\widetilde{Z_j} = \frac{Z_j - \mu_j}{\sqrt{\sigma^2 + \epsilon}}
$$

The addition of the extremely small quantity $\epsilon$ here is to avoid a zero denominator.

However, batch normalization has some flaws, for example:

- When GPU memory is limited and the mini-batch is small, the mean and variance that Batch Norm takes from the samples cannot reflect the global statistical distribution, degrading the results;
- For RNNs unrolled over the time dimension, the same-position distributions of different sentences are most likely different, so Batch Norm's normalization loses its meaning;
- During training, Batch Norm needs to save the statistics (mean and variance) of each step. At test time, due to the variable-length nature of sentences, the test set may contain longer sentences than the training set, so for later-position steps there are no trained statistics available;
- Applying Batch Norm requires saving and computing the batch statistics at every step, which is both time-consuming and laborious

Therefore, Layer Normalization (Layer Norm), which is more commonly used in deep neural networks and works better, emerged. Unlike Batch Norm, which computes the mean and variance over all samples at each layer, Layer Norm computes the mean and variance over all layers for each sample, thereby making each sample's distribution stable. Layer Norm's normalization approach is exactly the same as Batch Norm's; only the dimension over which the statistics are computed differs.

Based on the normalization formula above, we can simply implement a Layer Norm layer:

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
Note that in the Layer Norm layer we implemented above, there are two linear matrices performing the mapping.

### 2.2.4 Residual Connection

Since the Transformer's model structure is relatively complex and the number of layers is deep, to avoid model degradation, the Transformer adopts the idea of residual connections to link every sub-layer. A residual connection means that the input of the next layer is not only the output of the previous layer, but also includes the input of the previous layer. Residual connections allow information from the lowest layer to be passed directly to the highest layer, letting the higher layers focus on learning the residuals.

​For example, in the Encoder, in the first sub-layer, the input first undergoes Layer Normalization (Layer Norm), then enters the multi-head self-attention layer, and its output is added to the original input. The second sub-layer works the same way. That is:

$$
x = x + MultiHeadSelfAttention(LayerNorm(x))
$$

$$
output = x + FNN(LayerNorm(x))
$$

In the code implementation, we realize the residual connection by adding the original value in the layer's forward computation:

```python
# Attention computation
h = x + self.attention.forward(self.attention_norm(x))
# Pass through the feed forward neural network
out = h + self.feed_forward.forward(self.fnn_norm(h))
```

In the code above, self.attention_norm and self.fnn_norm are both LayerNorm layers, self.attn is the attention layer, and self.feed_forward is the feed forward neural network.

### 2.2.5 Encoder


After implementing the components above, we can build the Transformer's Encoder. The Encoder consists of N Encoder Layers, and each Encoder Layer includes one attention layer and one feed forward neural network. Therefore, we can first implement an Encoder Layer:

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

Then we build an Encoder, composed of N Encoder Layers, with a Layer Norm added at the end to achieve normalization:

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

Similarly, we can first build a Decoder Layer, and then assemble the N Decoder Layers into a Decoder. But unlike the Encoder, the Decoder consists of two attention layers and one feed forward neural network. The first attention layer is a masked self-attention layer, i.e., attention computation using a Mask, which ensures that each token can only use the attention scores of the tokens before it; the second attention layer is a multi-head attention layer, which uses the output of the first attention layer as the query, and the Encoder's output as the key and value, to compute the attention scores. Finally, it passes through the feed forward neural network:

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

Then, in the same way, we build a Decoder block:

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

With the above Encoder and Decoder built, the core part of the Transformer is complete. Next, concatenating the Encoder and Decoder and adding the Embedding layers yields the complete Transformer model.

## 2.3 Building a Transformer

In the previous two sections, we analyzed in depth the Attention mechanism and the core of the Transformer—the Encoder and Decoder structures. Next, based on the components implemented in the previous section, we can build a complete Transformer model.

### 2.3.1 The Embedding Layer

As we discussed in Chapter 1, in NLP tasks we often need to convert the natural language input into vectors that a machine can process. In deep learning, the component that undertakes this task is the Embedding layer.

The Embedding layer is essentially a lookup table of embedding vectors for a fixed-size vocabulary. That is, before feeding the input into the neural network, we usually first pass the natural language input through a tokenizer; the tokenizer's role is to split the natural language input into tokens and convert them into fixed indices. For example, if we set the vocabulary size to 4 and the input is “我喜欢你” (I like you), then the tokenizer can convert the input into:

```
input: 我 (I)
output: 0

input: 喜欢 (like)
output: 1

input：你 (you)
output: 2
```

Of course, in practice, the tokenizer's work is more complicated than this. For example, there are many different tokenization methods: splitting into words, splitting into subwords, splitting into characters, etc., and the vocabulary size often reaches tens or even hundreds of thousands. We will not elaborate on the tokenizer's details here; later on, we will introduce in detail how the tokenizer of a large model operates and is trained.

Therefore, the Embedding layer's input is usually a matrix of shape (batch_size, seq_len, 1): the first dimension is the number of samples processed in one batch, the second dimension is the length of the natural language sequence, and the third dimension is the index value to which each token was converted by the tokenizer. For example, for the input above, the Embedding layer's input would be:

```
[[[0],[1],[2]]]
```

Its batch_size is 1, its seq_len is 3, and the converted indices are as above.

Internally, the Embedding is actually a trainable (Vocab_size, embedding_dim) weight matrix; every entry in the vocabulary corresponds to one row—a vector of dimension embedding_dim. For an input value, the corresponding word vector is looked up, and then concatenated into a (batch_size, seq_len, embedding_dim) matrix as the output.

The above implementation is not complicated; we can directly use the Embedding layer from torch:

```python
self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
```

### 2.3.2 Positional Encoding

The attention mechanism achieves good parallel computation, but at the same time, its attention computation also causes the loss of relative positions in the sequence. In RNNs and LSTMs, the input sequence is recursively processed one step at a time along the sentence's own order, so the order of the input sequence carries extremely important information, which also fits well with the inherent nature of natural language.

However, from the analysis of the attention mechanism above, we can see that in the attention computation, for every token in the sequence, all other positions are equal to it; that is, “我喜欢你” (I like you) and “你喜欢我” (You like me) look completely identical to the attention mechanism—but this is undoubtedly a huge problem of the attention mechanism. Therefore, in order to use sequence order information and preserve the relative position information in the sequence, the Transformer adopts a positional encoding mechanism, which was later adopted by many models.

​Positional encoding means encoding each token according to its relative position in the sequence, and then adding the positional encoding to the word vector encoding. There are many ways to perform positional encoding; the Transformer uses sine and cosine functions for positional encoding (absolute positional encoding, Sinusoidal), encoded as follows:

$$
PE(pos, 2i) = sin(pos/10000^{2i/d_{model}})\\
PE(pos, 2i+1) = cos(pos/10000^{2i/d_{model}})
$$

In the formula above, pos is the token's position in the sentence, and 2i and 2i+1 indicate whether the positional encoding vector's dimension index is odd or even. From the formula above, we can see that for odd and even dimensions, the Transformer uses different functions for encoding.

Let us illustrate the positional encoding computation with a simple example: suppose the input is a sentence of length 4, "I like to code". We can obtain the following word vector matrix $\rm x$, where each row represents one word vector; $\rm x_0=[0.1,0.2,0.3,0.4]$ corresponds to the word vector of "I", whose pos is 0, and so on; the second row represents the word vector of "like", whose pos is 1:

$$
\rm x = \begin{bmatrix} 
0.1 & 0.2 & 0.3 & 0.4 \\   
0.2 & 0.3 & 0.4 & 0.5 \\    
0.3 & 0.4 & 0.5 & 0.6 \\    
0.4 & 0.5 & 0.6 & 0.7
\end{bmatrix}
$$

​Then the word vectors after positional encoding are:

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

We can use the following code to obtain the positional encoding of the example above:
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

1. It allows the PE to adapt to sentences longer than any in the training set: suppose the longest sentence in the training set has 20 words, and suddenly a sentence of length 21 arrives; using the formula-based computation method, the Embedding of the 21st position can be computed.
2. It allows the model to easily compute relative positions: for a fixed-length offset k, PE(pos+k) can be computed from PE(pos), because Sin(A+B) = Sin(A)Cos(B) + Cos(A)Sin(B), Cos(A+B) = Cos(A)Cos(B) - Sin(A)Sin(B).

We can also prove the superiority of this encoding method through rigorous mathematical derivation. The original Transformer Embedding can be expressed as:

$$
\begin{equation}f(\cdots,\boldsymbol{x}_m,\cdots,\boldsymbol{x}_n,\cdots)=f(\cdots,\boldsymbol{x}_n,\cdots,\boldsymbol{x}_m,\cdots)\end{equation}
$$

Clearly, such a function is not asymmetric, meaning it cannot represent relative position information. We want to obtain the following kind of encoding:

$$
\begin{equation}\tilde{f}(\cdots,\boldsymbol{x}_m,\cdots,\boldsymbol{x}_n,\cdots)=f(\cdots,\boldsymbol{x}_m + \boldsymbol{p}_m,\cdots,\boldsymbol{x}_n + \boldsymbol{p}_n,\cdots)\end{equation}
$$

Here, the added $p_m$ and $p_n$ are the positional encodings. Next, we Taylor-expand $f(...,x_m+p_m,...,x_n+p_n)$ at the two positions m and n:

$$
\begin{equation}\tilde{f}\approx f + \boldsymbol{p}_m^{\top} \frac{\partial f}{\partial \boldsymbol{x}_m} + \boldsymbol{p}_n^{\top} \frac{\partial f}{\partial \boldsymbol{x}_n} + \frac{1}{2}\boldsymbol{p}_m^{\top} \frac{\partial^2 f}{\partial \boldsymbol{x}_m^2}\boldsymbol{p}_m + \frac{1}{2}\boldsymbol{p}_n^{\top} \frac{\partial^2 f}{\partial \boldsymbol{x}_n^2}\boldsymbol{p}_n + \underbrace{\boldsymbol{p}_m^{\top} \frac{\partial^2 f}{\partial \boldsymbol{x}_m \partial \boldsymbol{x}_n}\boldsymbol{p}_n}_{\boldsymbol{p}_m^{\top} \boldsymbol{\mathcal{H}} \boldsymbol{p}_n}\end{equation}
$$

We can see that the 1st term is independent of position, terms 2–5 depend on only a single position, and the 6th term (where f is partially differentiated with respect to m and n respectively) involves both positions. Therefore, we hope the 6th term ($p_m^THp_n$) expresses the relative position information, i.e., we seek a function g such that:

$$
p_m^THp_n = g(m-n)
$$

Suppose $H$ is an identity matrix; then:

$$
p_m^THp_n = p_m^Tp_n = \langle\boldsymbol{p}_m, \boldsymbol{p}_n\rangle = g(m-n)
$$

By treating the vector [x,y] as the complex number x+yi, we construct equations based on the rules of complex number operations:

$$
\begin{equation}\langle\boldsymbol{p}_m, \boldsymbol{p}_n\rangle = \text{Re}[\boldsymbol{p}_m \boldsymbol{p}_n^*]\end{equation}
$$

Further suppose there exists a complex number $q_{m-n}$ such that:

$$
\begin{equation}\boldsymbol{p}_m \boldsymbol{p}_n^* = \boldsymbol{q}_{m-n}\end{equation}
$$

Solving this equation using the exponential form of complex numbers yields the solution of the positional encoding in the two-dimensional case:

$$
\begin{equation}\boldsymbol{p}_m = e^{\text{i}m\theta}\quad\Leftrightarrow\quad \boldsymbol{p}_m=\begin{pmatrix}\cos m\theta \\ \sin m\theta\end{pmatrix}\end{equation}
$$

Since the inner product satisfies linear superposition, the positional encoding of a higher-dimensional even dimension can be expressed as a combination of multiple two-dimensional positional encodings:

$$
\begin{equation}\boldsymbol{p}_m = \begin{pmatrix}e^{\text{i}m\theta_0} \\ e^{\text{i}m\theta_1} \\ \vdots \\ e^{\text{i}m\theta_{d/2-1}}\end{pmatrix}\quad\Leftrightarrow\quad \boldsymbol{p}_m=\begin{pmatrix}\cos m\theta_0 \\ \sin m\theta_0 \\ \cos m\theta_1 \\ \sin m\theta_1 \\ \vdots \\ \cos m\theta_{d/2-1} \\ \sin m\theta_{d/2-1}  \end{pmatrix}\end{equation}
$$

Then taking $\theta_i = 10000^{-2i/d}$ (this form ensures that as |m−n| grows, ⟨pm,pn⟩ exhibits a trend toward zero, which can be proven by integrating over the positional encodings, and the base of 10000 is an experimental result) yields the encoding method above.

When $H$ is not an identity matrix: since the correlation between any two of the d-dimensional vectors formed by the model's Embedding layer is relatively small, satisfying a certain degree of decoupling, we can regard it as a diagonal matrix; then using the above encoding:

$$
\begin{equation}\boldsymbol{p}_m^{\top} \boldsymbol{\mathcal{H}} \boldsymbol{p}_n=\sum_{i=1}^{d/2} \boldsymbol{\mathcal{H}}_{2i,2i} \cos m\theta_i \cos n\theta_i + \boldsymbol{\mathcal{H}}_{2i+1,2i+1} \sin m\theta_i \sin n\theta_i\end{equation}
$$

Through product-to-sum identities:

$$
\begin{equation}\sum_{i=1}^{d/2} \frac{1}{2}\left(\boldsymbol{\mathcal{H}}_{2i,2i} + \boldsymbol{\mathcal{H}}_{2i+1,2i+1}\right) \cos (m-n)\theta_i + \frac{1}{2}\left(\boldsymbol{\mathcal{H}}_{2i,2i} - \boldsymbol{\mathcal{H}}_{2i+1,2i+1}\right) \cos (m+n)\theta_i \end{equation}
$$

This shows that this encoding can still represent relative position.

The encoding results above are shown in Figure 2.6:

<div align="center">
  <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/2-figures/3-0.png" alt="Figure caption" width="90%"/>
  <p>Figure 2.6 Encoding Results</p>
</div>


Based on the above principle, we implement a positional encoding layer:

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

Concatenating all of the components above according to the Transformer structure in the figure below yields a complete Transformer model, as shown in Figure 2.7:

<div align="center">
  <img src="https://raw.githubusercontent.com/datawhalechina/happy-llm/main/docs/images/2-figures/3-1.png" alt="Figure caption" width="80%"/>
  <p>Figure 2.7 Transformer Model Structure</p>
</div>

But note that the figure above is the illustration from the original paper "Attention is all you need", in which the LayerNorm layer is placed after the Attention layer—the "Post-Norm" structure—but in its released source code, the LayerNorm layer is placed before the Attention layer, i.e., the "Pre Norm" structure. Considering that current LLMs generally adopt the "Pre-Norm" structure (which makes the loss more stable), this chapter adopts the "Pre-Norm" structure in its implementation.

As shown in the figure, the output after being mapped by the tokenizer first passes through the Embedding layer and the Positional Embedding layer for encoding, then enters the N Encoders and N Decoders described in the previous section (in the original Transformer model, N is set to 6), and finally passes through a linear layer and a Softmax layer to obtain the final output.

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

Note that besides building the entire Transformer structure, the code above additionally implements three functions:

- get_num_params: used to count the model's number of parameters
- _init_weights: used to randomly initialize all of the model's parameters
- forward: the forward computation function

In addition, in the forward computation function, we use PyTorch's cross-entropy function to compute the loss. For other loss functions, readers can consult PyTorch's official documentation; we will not elaborate on this here.

After the above steps, we can "hand-build" a complete, computable Transformer model from scratch. Since this book focuses mainly on LLMs, in this chapter we will not go into detail on how to train the Transformer model; in later chapters, we will similarly "hand-build" a LLaMA model from scratch and walk you step by step through training your own Tiny LLaMA.

**References**

[1] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin. (2023). *Attention Is All You Need.* arXiv preprint arXiv:1706.03762.

[2] Jay Mody's article "An Intuition for Attention". Source: https://jaykmody.com/blog/attention-intuition/
