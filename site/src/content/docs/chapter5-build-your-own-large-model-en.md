---
title: "Chapter 5: Build Your Own Large Model"
---



## 5.1 Implementing a LLaMA2 Large Model by Hand

Meta (formerly Facebook) released LLaMA, its first Transformer-based large language model, in February 2023, and released LLaMA2, a model in the same family, in July of the same year. In Chapter 4 we already learned about LLMs and how to train them. In this section, we will learn how to implement a LLaMA2 model by hand.

The LLaMA2 model architecture is shown in Figure 5.1 below:

<div align='center'>
    <img src="/images/5-images/LLama2.webp" alt="Figure 5.1 LLaMA2 Architecture" width="1600" height="1142" loading="lazy" decoding="async" />
    <p>Figure 5.1 LLaMA2 Architecture</p>
</div>

### 5.1.1 Defining the Hyperparameters

First, we need to define some hyperparameters. These hyperparameters include the model size, number of layers, number of heads, embedding dimension, hidden layer dimension, and so on. These hyperparameters can be adjusted according to the actual situation.

Here we define a custom `ModelConfig` class to store and record our hyperparameters. We inherit from the `PretrainedConfig` class, which is the parameter class in the `transformers` library. By inheriting from this class, we can conveniently use some features of the `transformers` library, and it also makes it easier to export Hugging Face models later.

```python
from transformers import PretrainedConfig

class ModelConfig(PretrainedConfig):
    model_type = "Tiny-K"
    def __init__(
            self,
            dim: int = 768, # Model dimension
            n_layers: int = 12, # Number of Transformer layers
            n_heads: int = 16, # Number of attention heads
            n_kv_heads: int = 8, # Number of key-value heads
            vocab_size: int = 6144, # Vocabulary size
            hidden_dim: int = None, # Hidden layer dimension
            multiple_of: int = 64, 
            norm_eps: float = 1e-5, # eps for the normalization layer
            max_seq_len: int = 512, # Maximum sequence length
            dropout: float = 0.0, # dropout probability
            flash_attn: bool = True, # Whether to use Flash Attention
            **kwargs,
    ):
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.flash_attn = flash_attn
        super().__init__(**kwargs)
```

> Whenever `args` appears in the code below, it refers by default to the above `ModelConfig` configuration.

Let's take a look at the meaning of some of these hyperparameters. For example, `dim` is the model dimension, `n_layers` is the number of Transformer layers, `n_heads` is the number of attention heads, `vocab_size` is the vocabulary size, `max_seq_len` is the maximum input sequence length, and so on. The code above also includes detailed comments for every parameter, and we will use these hyperparameters to build our model in the code that follows.

### 5.1.2 Building RMSNorm

`RMSNorm` can be expressed by the following mathematical formula:

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{n}\sum_{i=1}^{n}x_i^2 + \epsilon}} \cdot \gamma
$$

where:
- $x_i$ is the $i$-th element of the input vector
- $\gamma$ is a learnable scaling parameter (corresponding to `self.weight` in the code)
- $n$ is the dimension of the input vector
- $\epsilon$ is a small constant used for numerical stability (to avoid division by zero)

This normalization helps stabilize the learning process by ensuring that the scale of the weights does not become too large or too small, which is especially useful in deep learning models with many layers.

We can implement `RMSNorm` with the following code:

```python
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        # eps prevents division by zero
        self.eps = eps
        # weight is a learnable parameter, initialized to all ones
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # Core part of RMSNorm
        # x.pow(2).mean(-1, keepdim=True) computes the mean of the squares of input x
        # torch.rsqrt is the reciprocal of the square root, which gives the denominator of RMSNorm; adding eps prevents division by zero
        # Finally, multiply by x to obtain the RMSNorm result
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # The forward function is the model's forward pass
        # First convert input x to float, then apply RMSNorm, and finally convert back to the original dtype
        # Finally multiply by weight, a learnable scaling factor of RMSNorm
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
```

Furthermore, we can use the following code to test the `RMSNorm` module. The final output shape is `torch.Size([1, 50, 768])`, which is the same as our input shape, indicating that the module is implemented correctly and that normalization does not change the input shape.

```python
norm = RMSNorm(args.dim, args.norm_eps)
x = torch.randn(1, 50, args.dim)
output = norm(x)
print(output.shape)

out:
torch.Size([1, 50, 768])
```

### 5.1.3 Building LLaMA2 Attention

In the LLaMA2 model, although only the LLaMA2-70B model uses Grouped-Query Attention (GQA), we still choose to use GQA to build our LLaMA Attention module, which can improve the model's efficiency and save some GPU memory.

<div align='center'>
    <img src="/images/5-images/llama2-attention.webp" alt="Figure 5.2 LLaMA2 Attention Structure" width="1600" height="3128" loading="lazy" decoding="async" />
    <p>Figure 5.2 LLaMA2 Attention Structure</p>
</div>

#### 5.1.3.1 repeat_kv

In the LLaMA2 model, we need to expand the dimensions of the keys and values to match those of the queries so that the attention computation can be performed. We can implement `repeat_kv` with the following code:

```python
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    # Get the input tensor's shape: batch size, sequence length, number of key/value heads, head dimension
    bs, slen, n_kv_heads, head_dim = x.shape
    
    # If the repetition count is 1, no repetition is needed; return the original tensor directly
    if n_rep == 1:
        return x
    
    # Expand and reshape the tensor to repeat the key-value pairs
    return (
        x[:, :, :, None, :]  # Add a new dimension before the fourth dimension (the head dimension)
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)  # Expand the newly added dimension to n_rep to achieve the repetition effect
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)  # Reshape to merge the key/value head count and repetition dimensions
    )
```

In the above code:

- First, get the shape of the input tensor: the code obtains the input tensor's shape via x.shape, including the batch size (bs), sequence length (slen), number of key/value heads (n_kv_heads), and the per-head dimension (head_dim).

- Then, check the repetition count: next, the code checks whether the repetition count n_rep is 1. If it is 1, no repetition of the keys and values is needed, and the original tensor x is returned directly.

- Finally, expand and reshape the tensor:
  - Add a new dimension after the third dimension (i.e., the key/value head dimension), forming `x[:, :, :, None, :]`.
  - Use the `expand` method to expand the newly added dimension to `n_rep`, achieving the repetition of the key/value pairs.
  - Finally, reshape via the reshape method to merge the expanded dimension back into the key/value head count, i.e., `x.reshape(bs, slen, n_kv_heads * n_rep, head_dim)`, so that the final tensor shape matches the query dimension.

#### 5.1.3.2 Rotary Embeddings

Next we implement rotary embeddings. Rotary embeddings are an important component of the LLaMA2 model; they provide the attention mechanism with stronger contextual information, thereby improving model performance.

First, we construct the function that obtains the real and imaginary parts of the rotary embeddings:

```python
# Note: dim here should be dim//n_head, because we apply the rotary embedding to each head
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    # torch.arange(0, dim, 2)[: (dim // 2)].float() generates a sequence starting at 0 with step 2, of length dim//2
    # Then divide each element by dim, and take the reciprocal with theta as the exponent, to get the frequencies
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # Generate a sequence from 0 to end, of length end
    t = torch.arange(end, device=freqs.device)
    # Compute the outer product to get a 2D matrix, where each row is an element of t times an element of freqs
    freqs = torch.outer(t, freqs).float()
    # Take the cosine of the frequencies to get the real part
    freqs_cos = torch.cos(freqs)
    # Take the sine of the frequencies to get the imaginary part
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin
```

- Computing the frequency sequence:
  - `torch.arange(0, dim, 2)[: (dim // 2)].float()` generates a sequence starting at 0 with step 2, whose length is half of `dim`.
  - Dividing each element by `dim` and then taking the reciprocal with respect to `theta` yields a frequency sequence `freqs`. This step generates frequencies suitable for rotary embeddings.
- Generating the time sequence:
  - `t = torch.arange(end, device=freqs.device)` generates a sequence from `0` to `end`, of length `end`. `end` is usually the maximum sequence length.
- Computing the outer product of the frequencies
  - `freqs = torch.outer(t, freqs).float()` computes the outer product of the time sequence `t` and the frequency sequence `freqs`, yielding a 2D matrix `freqs`. Each row is an element of the time sequence `t` multiplied by an element of the frequency sequence `freqs`.
- Computing the real and imaginary parts
  - `freqs_cos = torch.cos(freqs)` takes the cosine of the frequency matrix `freqs`, giving the real part of the rotary embeddings.
  - `freqs_sin = torch.sin(freqs)` takes the sine of the frequency matrix `freqs`, giving the imaginary part of the rotary embeddings.

In the end, this function returns two matrices, `freqs_cos` and `freqs_sin`, representing the real and imaginary parts of the rotary embeddings respectively, which are used in subsequent computations.

Next, we construct the `reshape_for_broadcast` function for adjusting tensor shapes. The main purpose of this function is to adjust the shape of `freqs_cis` so that it aligns with the dimensions of `x` during broadcasting, allowing correct tensor operations.

```python
def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    # Get the number of dimensions of x
    ndim = x.ndim
    
    # Assert that 1 is within the dimension range of x
    assert 0 <= 1 < ndim
    
    # Assert that the shape of freqs_cis matches x's second dimension and last dimension
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    
    # Construct a new shape where all dimensions except the second and last are 1; this allows freqs_cis to be broadcast against x
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    
    # Reshape freqs_cis to the new shape and return it
    return freqs_cis.view(shape)
```

Finally, we can implement the rotary embeddings with the following code:

```python
def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:

    # Convert the query and key tensors to float and reshape to separate real and imaginary parts
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # Reshape the frequency tensors for broadcasting
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # Apply the rotation, computing the rotated real and imaginary parts respectively
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # Merge the last two dimensions and restore the original tensor shape
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)
```

Here we provide code for testing the `apply_rotary_emb` function; you can also try adding breakpoints in the code to inspect the computed results at each step.

```python
xq = torch.randn(1, 50, 6, 48) # bs, seq_len, dim//n_head, n_head_dim
xk = torch.randn(1, 50, 6, 48) # bs, seq_len, dim//n_head, n_head_dim

# Use the precompute_freqs_cis function to get sin and cos
cos, sin = precompute_freqs_cis(288//6, 50)
print(cos.shape, sin.shape)
xq_out, xk_out = apply_rotary_emb(xq, xk, cos, sin)

xq_out.shape, xk_out.shape
```

OUT:
```
torch.Size([50, 24]) torch.Size([50, 24])

(torch.Size([1, 50, 6, 48]), torch.Size([1, 50, 6, 48]))
```

#### 5.1.3.3 Assembling LLaMA2 Attention

Now that we have finished implementing the rotary embeddings, we can build the LLaMA2 Attention module.

```python
class Attention(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        # Determine the number of heads used for keys and values, depending on whether n_kv_heads is specified.
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # Ensure the total number of heads is divisible by the number of kv heads.
        assert args.n_heads % self.n_kv_heads == 0

        # Model parallel size, defaults to 1.
        model_parallel_size = 1
        # Number of local heads, equal to the total number of heads divided by the model parallel size.
        self.n_local_heads = args.n_heads // model_parallel_size
        # Number of local kv heads, equal to the kv head count divided by the model parallel size.
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        # Repetition count, used to expand the size of keys and values.
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        # Per-head dimension, equal to the model dimension divided by the total number of heads.
        self.head_dim = args.dim // args.n_heads

        # Define the weight matrices.
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        # Output weight matrix.
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        # Define dropout.
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        # Save the dropout probability.
        self.dropout = args.dropout

        # Check whether Flash Attention is available (requires PyTorch >= 2.0).
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            # If Flash Attention is not supported, use a manually implemented attention mechanism and set up a mask.
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # Create an upper-triangular matrix to mask out future information.
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            # Register it as a model buffer
            self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor):
        # Get the batch size and sequence length, [batch_size, seq_len, dim]
        bsz, seqlen, _ = x.shape

        # Compute queries (Q), keys (K), and values (V).
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        # Reshape to fit the head dimensions.
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # Apply rotary position embeddings (RoPE).
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # Expand the keys and values to match the repetition count.
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        # Treat the heads as a batch dimension.
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Choose the implementation depending on Flash Attention support.
        if self.flash:
            # Use Flash Attention.
            output = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            # Use the manually implemented attention mechanism.
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            assert hasattr(self, 'mask')
            scores = scores + self.mask[:, :, :seqlen, :seqlen]
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv)

        # Restore the time dimension and merge the heads.
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # Finally project back into the residual stream.
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output
```

Likewise, you can use the following code to test the attention module. The final output shape is `torch.Size([1, 50, 768])`, the same as our input shape, indicating that the module is implemented correctly.

```python
# Create an Attention instance
attention_model = Attention(args)

# Simulate input data
batch_size = 1
seq_len = 50  # Assume the actual sequence length used is 50
dim = args.dim
x = torch.rand(batch_size, seq_len, dim)  # Randomly generate the input tensor
# freqs_cos = torch.rand(seq_len, dim // 2)  # Simulate cos frequencies, used for RoPE
# freqs_sin = torch.rand(seq_len, dim // 2)  # Simulate sin frequencies, used for RoPE

freqs_cos, freqs_sin = precompute_freqs_cis(dim//args.n_heads, seq_len)

# Run the Attention model
output = attention_model(x, freqs_cos, freqs_sin)

# The shape after attention is still [batch_size, seq_len, dim]
print("Output shape:", output.shape)
```

OUT:
```
Output shape: torch.Size([1, 50, 768])
```

### 5.1.4 Building the LLaMA2 MLP Module

Compared with the LLaMA2 Attention module we implemented earlier, the LLaMA2 MLP module is simpler to implement. We can implement `MLP` with the following code:

```python
class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        # If the hidden layer dimension is not specified, set it to 4x the input dimension
        # Then reduce it to 2/3, and finally make sure it is a multiple of multiple_of
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        # Define the first linear layer, from input dimension to hidden dimension
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        # Define the second linear layer, from hidden dimension to input dimension
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        # Define the third linear layer, from input dimension to hidden dimension
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        # Define a dropout layer to prevent overfitting
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Forward pass
        # First, input x goes through the first linear layer and the SiLU activation function
        # Then, the result is multiplied by the result of input x through the third linear layer
        # Finally, pass through the second linear layer and the dropout layer
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
```

Let's take a close look at the `forward` function: first, the input `x` passes through the first linear layer `self.w1` and the `SILU` activation function; then, the result is multiplied by the result of the input `x` passing through the third linear layer `self.w3`; finally, it passes through the second linear layer `self.w2` and the `dropout` layer to produce the final output.

Likewise, you can use the following code to test the `LLaMAMLP` module. The final output shape is `torch.Size([1, 50, 768])`, the same as our input shape, indicating that the module is implemented correctly.

```python
# Create an MLP instance
mlp = MLP(args.dim, args.hidden_dim, args.multiple_of, args.dropout)
# Randomly generate data
x = torch.randn(1, 50, args.dim)
# Run the MLP model
output = mlp(x)
print(output.shape)
```

OUT:
```
torch.Size([1, 50, 768])
```

### 5.1.5 LLaMA2 Decoder Layer

At this point, we have implemented the `Attention` module and the `MLP` module of the `LLaMA2` model; next we can build the `LLaMA2` `Decoder Layer`.

```python
class DecoderLayer(nn.Module):
    def __init__(self, layer_id: int, args: ModelConfig):
        super().__init__()
        # Define the number of heads in multi-head attention
        self.n_heads = args.n_heads
        # Define the input dimension
        self.dim = args.dim
        # Define the per-head dimension, equal to the input dimension divided by the number of heads
        self.head_dim = args.dim // args.n_heads
        # Define a LLaMA2Attention object for multi-head attention computation
        self.attention = Attention(args)
        # Define a LLaMAMLP object for feedforward network computation
        self.feed_forward = MLP(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            multiple_of=args.multiple_of,
            dropout=args.dropout,
        )
        # Define the layer ID
        self.layer_id = layer_id
        # Define the normalization layer for the attention computation
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # Define the normalization layer for the feedforward network computation
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cos, freqs_sin):
        # Forward pass
        # First, input x passes through the attention normalization layer, then attention is computed, and the result is added to input x to get h
        # Then, h passes through the feedforward normalization layer, then the feedforward computation is performed, and the result is added to h to get the output
        h = x + self.attention.forward(self.attention_norm(x), freqs_cos, freqs_sin)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out
```

The `DecoderLayer` simply combines the `Attention` module and the `MLP` module we completed above, forming a complete `Transformer` module.

Likewise, you can use the following code to test the `DecoderLayer` module. The final output shape is `torch.Size([1, 50, 768])`, the same as our input shape, indicating that the module is implemented correctly.

```python
# Create a LLaMADecoderLayer instance
decoderlayer = DecoderLayer(0, args)

# Simulate input data
dim = args.dim
seq_len = 50

x = torch.randn(1, seq_len, dim) # [bs, seq_len, dim]

freqs_cos, freqs_sin = precompute_freqs_cis(dim//args.n_heads, seq_len)

out = decoderlayer(x, freqs_cos, freqs_sin)

print(out.shape) # Same shape as the input x: [batch_size, seq_len, dim]
```

OUT:
```
torch.Size([1, 50, 768])
```

### 5.1.6 Building the LLaMA2 Model

Great — we have completed the implementation of all the modules above, and now comes the exciting moment: we can build the `LLaMA2` model. The `LLaMA2` model simply stacks `DecoderLayer` modules together to form a complete `Transformer` model.

```python
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional
import math

class Transformer(PreTrainedModel):
    config_class = ModelConfig  # Configuration class
    last_loss: Optional[torch.Tensor] # Records the most recently computed loss

    def __init__(self, args: ModelConfig = None):
        super().__init__(args)
        # Initialize model parameters
        self.args = args
        # Vocabulary size
        self.vocab_size = args.vocab_size
        # Number of layers
        self.n_layers = args.n_layers

        # Token embedding layer
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        # Dropout layer
        self.dropout = nn.Dropout(args.dropout)
        # Decoder layers
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(DecoderLayer(layer_id, args))
        # Normalization layer
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        # Output layer
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        # Share the token embedding layer's weights with the output layer
        self.tok_embeddings.weight = self.output.weight 

        # Precompute the frequencies for the rotary position embeddings
        freqs_cos, freqs_sin = precompute_freqs_cis(self.args.dim // self.args.n_heads, self.args.max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # Initialize all weights
        self.apply(self._init_weights)
        # Special scaled initialization for the residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * args.n_layers))

        # Initialize the attribute holding the loss of the last forward pass
        self.last_loss = None
        self.OUT = CausalLMOutputWithPast()  # Output container
        self._no_split_modules = [name for name, _ in self.named_modules()]  # List of modules that must not be split

    def _init_weights(self, module):
        # Function for initializing weights
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
        - tokens: Optional[torch.Tensor], the input token tensor.
        - targets: Optional[torch.Tensor], the target token tensor.
        - kv_cache: bool, whether to use a key-value cache.
        - kwargs: other keyword arguments.

        - self.OUT: CausalLMOutputWithPast, contains the logits and loss.
        """

        if 'input_ids' in kwargs:
            tokens = kwargs['input_ids']
        if 'labels' in kwargs:
            targets = kwargs['labels']

        # Forward pass
        _bsz, seqlen = tokens.shape
        # Through the token embedding layer and dropout layer
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)
        # Get the frequencies for the rotary position embeddings
        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]

        # Through the decoder layers
        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin)
        # Through the normalization layer
        h = self.norm(h)

        if targets is not None:
            # If targets are given, compute the loss
            logits = self.output(h)
            self.last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0, reduction='none')
        else:
            # Small optimization at inference time: only forward the output at the last position
            logits = self.output(h[:, [-1], :]) 
            self.last_loss = None

        # Set the outputs
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('last_loss', self.last_loss)
        return self.OUT

    
    @torch.inference_mode()
    def generate(self, idx, stop_id=None, max_new_tokens=256, temperature=1.0, top_k=None):
        """
        Given an input sequence idx (a long tensor of shape (bz, seq_len)), complete the sequence by repeatedly generating new tokens.
        Runs in model.eval() mode. A less efficient sampling version that does not use a key/value cache.
        """
        index = idx.shape[1]
        for _ in range(max_new_tokens):
            # If the sequence context is too long, crop it to the maximum length
            idx_cond = idx if idx.size(1) <= self.args.max_seq_len else idx[:, -self.args.max_seq_len:]
            
            # Forward pass to get the logits at the last position in the sequence
            logits = self(idx_cond).logits
            logits = logits[:, -1, :] # Keep only the output of the last time step
            
            if temperature == 0.0:
                # Select the most likely index
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                # Scale the logits and apply softmax
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            

            if idx_next == stop_id:
                break

            # Append the sampled index to the sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx[:, index:] # Return only the generated tokens
```

Likewise, you can use the following code to test the `Transformer` module. The final output shape is `torch.Size([1, 1, 6144])`, consistent with our input, indicating that the module is implemented correctly.

```python
# LLaMA2Model.forward takes two arguments, tokens and targets, where tokens is the input tensor and should be of int type
x = torch.randint(0, 6144, (1, 50)) # [bs, seq_len]
# Instantiate the LLaMA2Model
model = Transformer(args=args)
# Count all parameters of the model
num_params = sum(p.numel() for p in model.parameters())
print('Number of parameters:', num_params)

out = model(x)
print(out.logits.shape) # [batch_size, 1, vocab_size]
```

OUT:
```
Number of parameters: 82594560
torch.Size([1, 1, 6144])
```
## 5.2 Training a Tokenizer

In natural language processing (NLP), a Tokenizer is a tool that breaks text into smaller units called tokens. These tokens can be words, subwords, characters, or even specific symbols. Tokenization is the first step in NLP and directly affects the quality of subsequent processing and analysis. Different types of tokenizers suit different application scenarios. Below are several common tokenizers and their characteristics.

### 5.2.1 Word-based Tokenizer

The **Word-based Tokenizer** is the simplest and most intuitive tokenization method. It splits text into words based on spaces and punctuation. The advantages of this method are its simplicity and directness, ease of implementation, and agreement with human intuition about language. However, it also has some obvious drawbacks, such as being unable to handle out-of-vocabulary (OOV) and rare words, and handling compound words (like "New York") or contractions (like "don't") insufficiently. In addition, the Word-based Tokenizer faces challenges when processing different languages, because some languages (such as Chinese and Japanese) have no explicit word separators.

Example:
```
Input: "Hello, world! There is Datawhale."
Output: ["Hello", ",", "world", "!", "There", "is", "Datawhale", "."]
```

In this example, the input sentence is split into a series of words and punctuation marks, with each word or punctuation mark treated as an independent token.

### 5.2.2 Character-based Tokenizer

The **Character-based Tokenizer** treats every character in the text as an independent token. This method handles text very finely, and is suitable for handling typos, OOV words, or new words. Since every character is an independent token, this method can capture extremely fine-grained linguistic features. This is particularly useful for certain application scenarios, such as generative tasks or tasks that must handle a large number of OOV words. However, this method also makes the token sequence very long, increasing the model's computational complexity and training time. Moreover, character-level splitting may lose some word-level semantic information, making it hard for the model to understand context.

Example:
```
Input: "Hello"
Output: ["H", "e", "l", "l", "o"]
```

In this example, the word "Hello" is split into individual characters, each treated as an independent token. This method can handle any language and character set, with great flexibility.

### 5.2.3 Subword Tokenizer

The **Subword Tokenizer** sits between words and characters, better balancing tokenization granularity with the ability to handle OOV words. The key idea of the Subword Tokenizer is to split text into units smaller than words but larger than characters, which both handles unknown words and preserves a certain amount of semantic information. Common subword tokenization methods include BPE, WordPiece, and Unigram.

#### (1) Byte Pair Encoding (BPE)

**BPE** is a statistics-based method that builds a subword vocabulary by repeatedly merging the most frequent character or character-sequence pairs. The advantages of this method are its simplicity and efficiency: it effectively handles unknown and rare words while keeping the vocabulary small. BPE's merging process is bottom-up, progressively merging the most frequent character pairs into new subwords until a predetermined vocabulary size is reached or no more high-frequency character pairs remain.

Example:
```
Input: "lower"
Output: ["low", "er"]

Input: "newest"
Output: ["new", "est"]
```

In this example, the word "lower" is split into the subwords "low" and "er", while "newest" is split into "new" and "est". This method effectively handles stems and affixes, preserving the word's basic semantic structure.

#### (2) WordPiece

**WordPiece** is another subword-based tokenization method, originally used in Google's BERT model. Similar to BPE, WordPiece generates its vocabulary by maximizing a likelihood function over subword sequences, but it pays more attention to language-model optimization when merging subwords. WordPiece prefers subwords that maximize the probability of the whole sentence, so that the tokenization result has a higher probability under a language model.

Example:
```
Input: "unhappiness"
Output: ["un", "##happiness"]
```

In this example, the word "unhappiness" is split into the subwords "un" and "##happiness", where "##" indicates this is a suffix subword. In this way, WordPiece handles compound and derived words better, preserving more semantic information.

#### (3) Unigram

The **Unigram** tokenization method is based on a probabilistic model and splits text by selecting subwords with the highest probability. The Unigram vocabulary is generated by training a language model, and can handle multiple languages and different types of text. The Unigram model assigns a probability to each subword, then performs the optimal segmentation according to these probabilities.

Example:
```
Input: "unhappiness"
Output: ["un", "happiness"]

Input: "newest"
Output: ["new", "est"]
```

In this example, the word "unhappiness" is split into the subwords "un" and "happiness", while "newest" is split into "new" and "est". This method handles subword segmentation effectively through a probabilistic model, making the segmentation results more consistent with language usage habits.

Each tokenization method has its own application scenarios, advantages, and disadvantages; choosing the right tokenizer is crucial to the success of NLP tasks.

### 5.2.4 Training a Tokenizer

Here we choose to use the BPE algorithm to train a Subword Tokenizer. BPE is a simple yet effective tokenization method that can handle OOV and rare words while keeping the vocabulary small. We will use Hugging Face's `tokenizers` library to train a BPE Tokenizer.


#### Step 1: Install and Import Dependencies

First, we need to install the `tokenizers` library. In addition, we need to install the `datasets` and `transformers` libraries for loading training data and for loading the trained Tokenizer.

```bash
pip install tokenizers datasets transformers
```

Then, import the required libraries.

```python
import random
import json
import os
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from tokenizers import (
    decoders,
    models,
    pre_tokenizers,
    trainers,
    Tokenizer,
)
from tokenizers.normalizers import NFKC
from typing import Generator
```

#### Step 2: Load Training Data

Here we use the same dataset as for pretraining (the Mobvoi Seq Monkey open-source dataset) to train the tokenizer. You can use `code/download_dataset.sh` and `code/deal_dataset.py` to download and preprocess the dataset.

> Note: Because the dataset is very large, training may run out of memory. Since this project is for learning purposes, we recommend manually splitting off a small portion of the dataset for training and validation. The author has also stored a trained tokenizer in the GitHub repository that you can use directly.

```python
def read_texts_from_jsonl(file_path: str) -> Generator[str, None, None]:
    """Read a JSONL file and safely extract text data"""
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                if 'text' not in data:
                    raise KeyError(f"Missing 'text' field in line {line_num}")
                yield data['text']
            except json.JSONDecodeError:
                print(f"Error decoding JSON in line {line_num}")
                continue
            except KeyError as e:
                print(e)
                continue
```

#### Step 3: Create the Configuration Files

Before training the BPE Tokenizer, we need to create a complete `Tokenizer` configuration, including `tokenizer_config.json` and `special_tokens_map.json`. These configuration files define the `Tokenizer`'s parameters and special tokens, which are used for training and loading the `Tokenizer`. Here our `chat_template` is kept consistent with the `Qwen2.5` model.

```python
def create_tokenizer_config(save_dir: str) -> None:
    """Create the complete tokenizer configuration files"""
    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False,
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "pad_token": "<|im_end|>",
        "unk_token": "<unk>",
        "model_max_length": 1000000000000000019884624838656,
        "clean_up_tokenization_spaces": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "chat_template": (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
            "{% elif message['role'] == 'user' %}"
            "<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
            "{% elif message['role'] == 'assistant' %}"
            "<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\n' }}"
            "{% endif %}"
        )
    }

    # Save the main configuration file
    with open(os.path.join(save_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    # Create special_tokens_map.json
    special_tokens_map = {
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "unk_token": "<unk>",
        "pad_token": "<|im_end|>",
        "additional_special_tokens": ["<s>", "</s>"]
    }
    with open(os.path.join(save_dir, "special_tokens_map.json"), "w", encoding="utf-8") as f:
        json.dump(special_tokens_map, f, ensure_ascii=False, indent=4)
```

#### Step 4: Train the BPE Tokenizer

Before training the BPE Tokenizer, we need to define a training function that trains the Tokenizer and saves the trained Tokenizer files. Here we use the `Tokenizer` class from the `tokenizers` library to train the BPE Tokenizer.

You can see that when training the Tokenizer, we configure some special tokens, such as `<unk>`, `<s>`, `</s>`, `<|im_start|>`, and `<|im_end|>`. These tokens are used to mark unknown words, the start and end of sentences, and the start and end of conversations. These special tokens help the model better understand text data and improve the model's generalization and effectiveness.

```python
def train_tokenizer(data_path: str, save_dir: str, vocab_size: int = 8192) -> None:
    """Train and save a custom tokenizer"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize the tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = NFKC()  # Add text normalization
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    # Configure special tokens
    special_tokens = [
        "<unk>", 
        "<s>", 
        "</s>", 
        "<|im_start|>", 
        "<|im_end|>"
    ]

    # Configure the trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=2,  # Raise the low-frequency word filtering
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    # Train the tokenizer
    print(f"Training tokenizer with data from {data_path}")
    texts = read_texts_from_jsonl(data_path)
    tokenizer.train_from_iterator(texts, trainer=trainer, length=os.path.getsize(data_path))

    # Verify the special token mappings
    try:
        assert tokenizer.token_to_id("<unk>") == 0
        assert tokenizer.token_to_id("<s>") == 1
        assert tokenizer.token_to_id("</s>") == 2
        assert tokenizer.token_to_id("<|im_start|>") == 3
        assert tokenizer.token_to_id("<|im_end|>") == 4
    except AssertionError as e:
        print("Special tokens mapping error:", e)
        raise

    # Save the tokenizer files
    tokenizer.save(os.path.join(save_dir, "tokenizer.json"))
    
    # Create the configuration files
    create_tokenizer_config(save_dir)
    print(f"Tokenizer saved to {save_dir}")
```


#### Step 5: Use the Trained Tokenizer

We can use the trained Tokenizer to process text data, such as encoding, decoding, generating conversations, and so on. Below is a simple example showing how to use the trained Tokenizer to process text data.

```python
def eval_tokenizer(tokenizer_path: str) -> None:
    """Evaluate the tokenizer's functionality"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    # Test basic properties
    print("\n=== Tokenizer Basic Info ===")
    print(f"Vocab size: {len(tokenizer)}")
    print(f"Special tokens: {tokenizer.all_special_tokens}")
    print(f"Special token IDs: {tokenizer.all_special_ids}")

    # Test the chat template
    messages = [
        {"role": "system", "content": "你是一个AI助手。"},  # ("You are an AI assistant.")
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "I'm fine, thank you. and you?"},
        {"role": "user", "content": "I'm good too."},
        {"role": "assistant", "content": "That's great to hear!"},
    ]
    
    print("\n=== Chat Template Test ===")
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        # add_generation_prompt=True
    )
    print("Generated prompt:\n", prompt, sep="")

    # Test encoding and decoding
    print("\n=== Encode/Decode Test ===")
    encoded = tokenizer(prompt, truncation=True, max_length=256)
    decoded = tokenizer.decode(encoded["input_ids"], skip_special_tokens=False)
    print("Decoded text matches original:", decoded == prompt)

    # Test special token handling
    print("\n=== Special Token Handling ===")
    test_text = "<|im_start|>user\nHello<|im_end|>"
    encoded = tokenizer(test_text).input_ids
    decoded = tokenizer.decode(encoded)
    print(f"Original: {test_text}")
    print(f"Decoded:  {decoded}")
    print("Special tokens preserved:", decoded == test_text)
```

```python
eval_tokenizer('your tokenizer path')
```

OUT:
```
=== Tokenizer基本信息 === (Tokenizer Basic Info)
Vocab size: 6144
Special tokens: ['<|im_start|>', '<|im_end|>', '<unk>', '<s>', '</s>']
Special token IDs: [3, 4, 0, 1, 2]

=== 聊天模板测试 === (Chat Template Test)
Generated prompt:
<|im_start|>system
你是一个AI助手。<|im_end|> (You are an AI assistant.)
<|im_start|>user
How are you?<|im_end|>
<|im_start|>assistant
I'm fine, thank you. and you?<|im_end|>
<|im_start|>user
I'm good too.<|im_end|>
<|im_start|>assistant
That's great to hear!<|im_end|>


=== 编码解码测试 === (Encode/Decode Test)
Decoded text matches original: False

=== 特殊token处理 === (Special Token Handling)
Original: <|im_start|>user
Hello<|im_end|>
Decoded:  <|im_start|> user
Hello<|im_end|>
Special tokens preserved: False
```
## 5.3 Pretraining a Small LLM

In the previous chapters, we became familiar with the model architectures of various large models and with how to train a Tokenizer. In this section, we will train an 80-million-parameter LLM by hand.

### 5.3.1 Data Download

First, we need to download the pretraining dataset. Here we use two open-source datasets that contain a large amount of Chinese conversation data and can be used to train a dialogue generation model.

- Mobvoi Seq Monkey open-source dataset: The Mobvoi Seq Monkey general text dataset is a large-language-model pretraining corpus formed by aggregating and cleaning a variety of publicly available data sources, including web pages, encyclopedias, blogs, Q&A, open-source code, books, newspapers and periodicals, patents, textbooks, and exam questions. The total size is roughly 10B tokens.

- BelleGroup: A Chinese dialogue dataset of 3.5 million entries, containing human-machine dialogues, person-to-person dialogues, and other types of dialogue data; it can be used to train a dialogue generation model.


```python
# Download the pretraining dataset
os.system("modelscope download --dataset ddzhu123/seq-monkey mobvoi_seq_monkey_general_open_corpus.jsonl.tar.bz2 --local_dir your_local_dir")
# Extract the pretraining dataset
os.system("tar -xvf your_local_dir/mobvoi_seq_monkey_general_open_corpus.jsonl.tar.bz2")

# Download the SFT dataset
os.system(f'huggingface-cli download --repo-type dataset --resume-download BelleGroup/train_3.5M_CN --local-dir BelleGroup')



# 1. Process the pretraining data
def split_text(text, chunk_size=512):
    """Split the text into chunks of a specified length"""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

input_file = 'mobvoi_seq_monkey_general_open_corpus.jsonl'

with open('seq_monkey_datawhale.jsonl', 'a', encoding='utf-8') as pretrain:
    with open(input_file, 'r', encoding='utf-8') as f:
        data = f.readlines()
        for line in tqdm(data, desc=f"Processing lines in {input_file}", leave=False):  # add a line-level progress bar
            line = json.loads(line)
            text = line['text']
            chunks = split_text(text)
            for chunk in chunks:
                pretrain.write(json.dumps({'text': chunk}, ensure_ascii=False) + '\n')

# 2. Process the SFT data

def convert_message(data):
    """
    Convert the raw data into the standard format
    """
    message = [
        {"role": "system", "content": "你是一个AI助手"},  # ("You are an AI assistant")
    ]
    for item in data:
        if item['from'] == 'human':
            message.append({'role': 'user', 'content': item['value']})
        elif item['from'] == 'assistant':
            message.append({'role': 'assistant', 'content': item['value']})
    return message

with open('BelleGroup_sft.jsonl', 'a', encoding='utf-8') as sft:
    with open('BelleGroup/train_3.5M_CN.json', 'r', encoding='utf-8') as f:
        data = f.readlines()
        for item in tqdm(data, desc="Processing", unit="lines"):
            item = json.loads(item)
            message = convert_message(item['conversations'])
            sft.write(json.dumps(message, ensure_ascii=False) + '\n')
```

### 5.3.2 Training the Tokenizer

First, we need to train a Tokenizer for text processing. The Tokenizer's role is to convert text into sequences of numbers so that the model can understand and process them. The dataset we use is the [Mobvoi Seq Monkey open-source dataset](https://www.modelscope.cn/datasets/ddzhu123/seq-monkey/files), which contains a large amount of Chinese text data and can be used to train the Tokenizer.

> Note: Because the dataset is large, training on your own local computer will be rather slow, so here we provide a Tokenizer that has already been trained, which you can use directly. If you want to train one yourself, refer to the code below.

```bash
python code/train_tokenizer.py
```

```python
import random
import json
import os
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from tokenizers import (
    decoders,
    models,
    pre_tokenizers,
    trainers,
    Tokenizer,
)
from tokenizers.normalizers import NFKC
from typing import Generator

random.seed(42)

def read_texts_from_jsonl(file_path: str) -> Generator[str, None, None]:
    """Read a JSONL file and safely extract text data"""
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                if 'text' not in data:
                    raise KeyError(f"Missing 'text' field in line {line_num}")
                yield data['text']
            except json.JSONDecodeError:
                print(f"Error decoding JSON in line {line_num}")
                continue
            except KeyError as e:
                print(e)
                continue

def create_tokenizer_config(save_dir: str) -> None:
    """Create the complete tokenizer configuration files"""
    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": True,
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "pad_token": "<|im_end|>",
        "unk_token": "<unk>",
        "model_max_length": 1000000000000000019884624838656,
        "clean_up_tokenization_spaces": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "chat_template": (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
            "{% elif message['role'] == 'user' %}"
            "<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
            "{% elif message['role'] == 'assistant' %}"
            "<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\n' }}"
            "{% endif %}"
        )
    }

    # Save the main configuration file
    with open(os.path.join(save_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    # Create special_tokens_map.json
    special_tokens_map = {
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "unk_token": "<unk>",
        "pad_token": "<|im_end|>",
        "additional_special_tokens": ["<s>", "</s>"]
    }
    with open(os.path.join(save_dir, "special_tokens_map.json"), "w", encoding="utf-8") as f:
        json.dump(special_tokens_map, f, ensure_ascii=False, indent=4)

def train_tokenizer(data_path: str, save_dir: str, vocab_size: int = 8192) -> None:
    """Train and save a custom tokenizer"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize the tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = NFKC()  # Add text normalization
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    # Configure special tokens
    special_tokens = [
        "<unk>", 
        "<s>", 
        "</s>", 
        "<|im_start|>", 
        "<|im_end|>"
    ]

    # Configure the trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=2,  # Raise the low-frequency word filtering
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    # Train the tokenizer
    print(f"Training tokenizer with data from {data_path}")
    texts = read_texts_from_jsonl(data_path)
    tokenizer.train_from_iterator(texts, trainer=trainer, length=os.path.getsize(data_path))

    # Verify the special token mappings
    try:
        assert tokenizer.token_to_id("<unk>") == 0
        assert tokenizer.token_to_id("<s>") == 1
        assert tokenizer.token_to_id("</s>") == 2
        assert tokenizer.token_to_id("<|im_start|>") == 3
        assert tokenizer.token_to_id("<|im_end|>") == 4
    except AssertionError as e:
        print("Special tokens mapping error:", e)
        raise

    # Save the tokenizer files
    tokenizer.save(os.path.join(save_dir, "tokenizer.json"))
    
    # Create the configuration files
    create_tokenizer_config(save_dir)
    print(f"Tokenizer saved to {save_dir}")

def eval_tokenizer(tokenizer_path: str) -> None:
    """Evaluate the tokenizer's functionality"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    # Test basic properties
    print("\n=== Tokenizer Basic Info ===")
    print(f"Vocab size: {len(tokenizer)}")
    print(f"Special tokens: {tokenizer.all_special_tokens}")
    print(f"Special token IDs: {tokenizer.all_special_ids}")

    # Test the chat template
    messages = [
        {"role": "system", "content": "你是一个AI助手。"},  # ("You are an AI assistant.")
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "I'm fine, thank you. and you?"},
        {"role": "user", "content": "I'm good too."},
        {"role": "assistant", "content": "That's great to hear!"},
    ]
    
    print("\n=== Chat Template Test ===")
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        # add_generation_prompt=True
    )
    print("Generated prompt:\n", prompt, sep="")

    # Test encoding and decoding
    print("\n=== Encode/Decode Test ===")
    encoded = tokenizer(prompt, truncation=True, max_length=256)
    decoded = tokenizer.decode(encoded["input_ids"], skip_special_tokens=False)
    print("Decoded text matches original:", decoded == prompt)

    # Test special token handling
    print("\n=== Special Token Handling ===")
    test_text = "<|im_start|>user\nHello<|im_end|>"
    encoded = tokenizer(test_text).input_ids
    decoded = tokenizer.decode(encoded)
    print(f"Original: {test_text}")
    print(f"Decoded:  {decoded}")
    print("Special tokens preserved:", decoded == test_text)

def main():
    # Configure paths
    data_path = "your data path"
    save_dir = "tokenizer_k"

    # Train the tokenizer
    train_tokenizer(
        data_path=data_path,
        save_dir=save_dir,
        vocab_size=6144
    )

    # Evaluate the tokenizer
    eval_tokenizer(save_dir)

if __name__ == '__main__':
    main()
```

After training is complete, you can use `eval_tokenizer()` to test the Tokenizer's functionality and make sure it works properly. In this function, we first load the trained Tokenizer, then test the Tokenizer's basic properties, chat template, encoding/decoding, and other functions. These tests help us verify the Tokenizer's correctness and ensure it works properly. The correct output is:

OUT:
```
=== Tokenizer基本信息 === (Tokenizer Basic Info)
Vocab size: 6144
Special tokens: ['<|im_start|>', '<|im_end|>', '<unk>', '<s>', '</s>']
Special token IDs: [3, 4, 0, 1, 2]

=== 聊天模板测试 === (Chat Template Test)
Generated prompt:
<|im_start|>system
你是一个AI助手。<|im_end|> (You are an AI assistant.)
<|im_start|>user
How are you?<|im_end|>
<|im_start|>assistant
I'm fine, thank you. and you?<|im_end|>
<|im_start|>user
I'm good too.<|im_end|>
<|im_start|>assistant
That's great to hear!<|im_end|>


=== 编码解码测试 === (Encode/Decode Test)
Decoded text matches original: False

=== 特殊token处理 === (Special Token Handling)
Original: <|im_start|>user
Hello<|im_end|>
Decoded:  <|im_start|> user
Hello<|im_end|>
Special tokens preserved: False
```
### 5.3.3 Dataset

#### PretrainDataset

Before the data is fed into the model, we still need to do some processing to convert the text data into Tokens the model can understand. Here we use PyTorch's Dataset class to load the dataset. We define a `PretrainDataset` class to load the preprocessed dataset. We inherit from `torch.utils.data.IterableDataset` to define this dataset, which allows us to process data more flexibly and efficiently.

```python
from torch.utils.data import Dataset

class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        # Precompute the starting byte offset of each line
        self._offsets = []
        with open(data_path, 'rb') as f:
            self._offsets.append(0)
            while f.readline():
                self._offsets.append(f.tell())
        self._total_lines = len(self._offsets) - 1  # the last tell() is EOF

    def __len__(self):
        return self._total_lines

    def __getitem__(self, index: int):
        with open(self.data_path, 'rb') as f:
            f.seek(self._offsets[index])
            line = f.readline().decode('utf-8')
        sample = json.loads(line)
        text = f"{self.tokenizer.bos_token}{sample['text']}"
        input_id = self.tokenizer(text).data['input_ids'][:self.max_length]
        text_len = len(input_id)
        # The remaining part not yet filled to the maximum length
        padding_len = self.max_length - text_len
        input_id = input_id + [self.padding] * padding_len
        # 0 means the loss is not computed
        loss_mask = [1] * text_len + [0] * padding_len

        input_id = np.array(input_id)
        X = np.array(input_id[:-1]).astype(np.int64)
        Y = np.array(input_id[1:]).astype(np.int64)
        loss_mask = np.array(loss_mask[1:]).astype(np.int64)
        return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(loss_mask)
```

As can be seen from the above code and Figure 5.3, the `Pretrain Dataset` mainly converts the `text` into `input_id` via the `tokenizer`, then splits the `input_id` into `X` and `Y`, where `X` consists of the first n-1 elements of `input_id` and `Y` consists of the last n-1 elements. The `loss_mask` is mainly used to mark which positions the loss should be computed on and which should be skipped.

<div align='center'>
    <img src="/images/5-images/pretrain_dataset.webp" alt="Figure 5.3 Pretraining Loss Computation" width="1030" height="464" loading="lazy" decoding="async" />
    <p>Figure 5.3 Pretraining Loss Computation</p>
</div>

The example in the figure shows the processing when `max_length=9`:
- **Input sequence**: `[BOS, T1, T2, T3, T4, T5, T6, T7, EOS]`
- **Sample split**:
  - X: `[BOS, T1, T2, T3, T4, T5, T6, T7]` → the model's input context
  - Y: `[T1, T2, T3, T4, T5, T6, T7, EOS]` → the model's prediction targets
- **Loss mask**:
  - Valid positions: `[0, 1, 1, 1, 1, 1, 1, 1, 1]` → loss computed only for T1-EOS

#### SFTDataset

The `SFTDataset` is actually a multi-turn dialogue dataset. Our goal is to teach the model how to conduct multi-turn conversations. At this stage, the input is the previous turn of the conversation, and the output is the current turn of the conversation.

```python
class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self._offsets = []
        with open(data_path, 'rb') as f:
            self._offsets.append(0)
            while f.readline():
                self._offsets.append(f.tell())
        self._total_lines = len(self._offsets) - 1 

    def __len__(self):
        return self._total_lines

    def generate_loss_mask(self, input_ids):
        # Generate the loss mask: 0 means the loss is not computed, 1 means it is computed
        mask = [0] * len(input_ids)
        a_sequence = self.tokenizer("<|im_start|>assistant\n")['input_ids']  # <|im_start|>assistant\n
        a_length = len(a_sequence)
        n = len(input_ids)
        i = 0
        
        while i <= n - a_length:
            # Check whether the current position matches the target subsequence
            match = True
            for k in range(a_length):
                if input_ids[i + k] != a_sequence[k]:
                    match = False
                    break
            if match:
                # Starting from the end of the subsequence, find the first 4 (eos_token_id)
                j = None
                for idx in range(i + a_length, n):
                    if input_ids[idx] == self.tokenizer.eos_token_id:
                        j = idx
                        break
                if j is not None:
                    start = i + a_length
                    end = j  # set the end position to j (inclusive of 4)
                    # Mark the interval as 1 (from start to end)
                    if start <= end:
                        for pos in range(start, end + 1):
                            if pos < len(mask):
                                mask[pos] = 1
                # Skip past the current subsequence to avoid overlapping matches
                i += a_length
            else:
                i += 1
        return mask

    def __getitem__(self, index: int):
        with open(self.data_path, 'rb') as f:
            f.seek(self._offsets[index])
            line = f.readline().decode('utf-8')
        sample = json.loads(line)
        text = self.tokenizer.apply_chat_template(sample, tokenize=False, add_generation_prompt=False)
        input_id = self.tokenizer(text).data['input_ids'][:self.max_length]
        text_len = len(input_id)
        # The remaining part not yet filled to the maximum length
        padding_len = self.max_length - text_len
        input_id = input_id + [self.padding] * padding_len
        # 0 means the loss is not computed
        loss_mask = self.generate_loss_mask(input_id)

        input_id = np.array(input_id)
        X = np.array(input_id[:-1]).astype(np.int64)
        Y = np.array(input_id[1:]).astype(np.int64)
        loss_mask = np.array(loss_mask[1:]).astype(np.int64)
        return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(loss_mask)
```

In the SFT stage, since we use a multi-turn dialogue dataset, we need to distinguish which positions the loss should be computed on and which should be skipped. In the code above, I use a `generate_loss_mask` function to generate the `loss_mask`. This function mainly generates the `loss_mask`; the generation rule is: when it encounters `|<im_start|>assistant\n`, it starts computing the loss until it reaches `|<im_end|>`. This ensures that the model only computes the loss on the current turn's dialogue content during the SFT stage, as shown in Figure 5.4.

<div align='center'>
    <img src="/images/5-images/sftdataset.webp" alt="Figure 5.4 SFT Loss Computation" width="1029" height="451" loading="lazy" decoding="async" />
    <p>Figure 5.4 SFT Loss Computation</p>
</div>

As we can see, the `X` and `Y` of the SFT Dataset are the same as those of the Pretrain Dataset; the only difference is that in the SFT Dataset we need to generate a `loss_mask` to mark which positions the loss should be computed on and which should be skipped. In the figure, the blue small squares in `Input ids` are the AI's answers, so those are the places the model needs to learn. Therefore, in the `loss_mask`, the positions corresponding to the blue small squares are yellow, and the other positions are gray. In the code, positions marked 1 in the `loss_mask` have the loss computed, and positions marked 0 do not.


### 5.3.4 Pretraining

Once the data preprocessing is done, we can start training the model. The model we use is a Decoder-only Transformer model with the same structure as LLaMA2, implemented in PyTorch. The relevant code is in the `code/k_model.py` file, which will not be repeated here: the source code contains detailed Chinese comments, and we gave a detailed introduction in the previous sections.

Regarding the model, one thing worth looking at closely is how a generative model implements token generation; see the `generate` method in the `Transforerm` class in the `k_model.py` file.

```python
@torch.inference_mode()
    def generate(self, idx, stop_id=None, max_new_tokens=256, temperature=1.0, top_k=None):
        """
        Given an input sequence idx (a long tensor of shape (bz, seq_len)), complete the sequence by repeatedly generating new tokens.
        Runs in model.eval() mode. A less efficient sampling version that does not use a key/value cache.
        """
        index = idx.shape[1]
        for _ in range(max_new_tokens):
            # If the sequence context is too long, crop it to the maximum length
            idx_cond = idx if idx.size(1) <= self.args.max_seq_len else idx[:, -self.args.max_seq_len:]
            
            # Forward pass to get the logits at the last position in the sequence
            logits = self(idx_cond).logits
            logits = logits[:, -1, :] # Keep only the output of the last time step
            
            if temperature == 0.0:
                # Select the most likely index
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                # Scale the logits and apply softmax
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            

            if idx_next == stop_id:
                break

            # Append the sampled index to the sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx[:, index:] # Return only the generated tokens

```

In the `generate` method, we first obtain the `logits` at the last position in the sequence, then generate a new `token` based on these `logits`. The newly generated `token` is then appended to the sequence, after which the model continues to generate the next `token`. Through this iterative process, we can generate complete text.

Next comes the most important part: training the model!

> Note: When using the code below to train the model, you need to set the `--data_path` argument to the path of the preprocessed dataset, e.g. `--data_path seq_monkey_datawhale.jsonl`, and also specify which GPUs to train on, e.g. `--gpus 0,1`.

```python
def get_lr(it, all):
    """
    Computes the learning rate for the current iteration, using a cosine annealing schedule

    Learning-rate schedule:
    1. Warmup phase: the learning rate grows linearly from 0 to the target learning rate
    2. Cosine annealing phase: the learning rate decays following a cosine function to the minimum learning rate
    3. Beyond the training steps: keep the minimum learning rate

    Args:
        it (int): the current iteration step
        all (int): the total number of iteration steps

    Returns:
        float: the learning rate corresponding to the current step
    """
    warmup_iters = args.warmup_iters  # number of warmup iterations
    lr_decay_iters = all  # total number of iterations over which the learning rate decays
    min_lr = args.learning_rate / 10  # minimum learning rate, 1/10 of the initial learning rate

    # Warmup phase: linear growth
    if it < warmup_iters:
        return args.learning_rate * it / warmup_iters
    
    # Beyond the training steps: keep the minimum learning rate
    if it > lr_decay_iters:
        return min_lr
    
    # Cosine annealing phase
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # cosine coefficient
    return min_lr + coeff * (args.learning_rate - min_lr)

def train_epoch(epoch):
    """
    Trains one epoch

    Implements the full training loop, including:
    1. Data loading and device transfer
    2. Dynamic learning-rate adjustment
    3. Forward pass and loss computation
    4. Gradient accumulation and backpropagation
    5. Gradient clipping and optimizer update
    6. Logging and model saving

    Args:
        epoch (int): the current epoch number
    """
    start_time = time.time()  # record the start time
    
    # Iterate over each batch in the data loader
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        # Move the data to the specified device (GPU/CPU)
        X = X.to(args.device)  # input sequence
        Y = Y.to(args.device)  # target sequence
        loss_mask = loss_mask.to(args.device)  # loss mask, used to ignore padding tokens

        # Compute the learning rate for the current step
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch)
        # Update the learning rate of all parameter groups in the optimizer
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Use the mixed-precision training context
        with ctx:
            # Forward pass
            out = model(X, Y)
            # Compute the loss and divide by the accumulation steps (for gradient accumulation)
            loss = out.last_loss / args.accumulation_steps
            # Flatten the loss mask to 1D
            loss_mask = loss_mask.view(-1)
            # Apply the mask to compute the valid loss (ignoring padding positions)
            loss = torch.sum(loss * loss_mask) / loss_mask.sum()

        # Use the scaler for mixed-precision backpropagation
        scaler.scale(loss).backward()

        # Perform an optimizer update every accumulation_steps steps
        if (step + 1) % args.accumulation_steps == 0:
            # Unscale the gradients to prepare for gradient clipping
            scaler.unscale_(optimizer)
            # Clip gradients to prevent gradient explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # Perform the optimizer step
            scaler.step(optimizer)
            # Update the scaler's scale factor
            scaler.update()

            # Zero the gradients; set_to_none=True saves memory
            optimizer.zero_grad(set_to_none=True)

        # Log every log_interval steps
        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            # Print the training progress information
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.7f} epoch_Time:{}min;'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,  # restore the true loss value
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))
            
            # If SwanLab is enabled, log the training metrics
            if args.use_swanlab:
                swanlab.log({
                    "loss": loss.item() * args.accumulation_steps,
                    "lr": optimizer.param_groups[-1]['lr']
                })

        # Save the model every save_interval steps
        if (step + 1) % args.save_interval == 0:
            model.eval()  # switch to evaluation mode
            # Build the checkpoint filename
            ckp = f'{args.save_dir}/pretrain_{lm_config.dim}_{lm_config.n_layers}_{lm_config.vocab_size}.pth'

            # Handle multi-GPU saving: if the model is a DataParallel model, access the .module attribute
            state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            torch.save(state_dict, ckp)
            model.train()  # switch back to training mode
        
        # Every 20000 steps, save a checkpoint tagged with the step count
        if (step + 1) % 20000 == 0:
            model.eval()
            # Build the step-tagged checkpoint filename
            ckp = f'{args.save_dir}/pretrain_{lm_config.dim}_{lm_config.n_layers}_{lm_config.vocab_size}_step{step+1}.pth'

            # Save the model state dict
            state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            torch.save(state_dict, ckp)
            model.train()


def init_model():
    """
    Initializes the model and tokenizer

    Functions include:
    1. Loading the pretrained tokenizer
    2. Creating the Transformer model
    3. Setting up multi-GPU parallel training (if available)
    4. Moving the model to the specified device
    5. Counting and printing the number of model parameters

    Returns:
        tuple: (model, tokenizer), the initialized model and tokenizer
    """
    def count_parameters(model):
        """
        Counts the number of trainable parameters in the model

        Args:
            model: the PyTorch model

        Returns:
            int: the total number of trainable parameters
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Load the pretrained tokenizer from a local path
    tokenizer = AutoTokenizer.from_pretrained('./tokenizer_k/')

    # Create the Transformer model from the configuration
    model = Transformer(lm_config)
    
    # Multi-GPU initialization: check the number of available GPUs and set up DataParallel
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        Logger(f"Using {num_gpus} GPUs with DataParallel!")
        # Wrap the model with DataParallel to support multi-GPU training
        model = torch.nn.DataParallel(model)
    
    # Move the model to the specified device (GPU or CPU)
    model = model.to(args.device)
    
    # Count and print the number of model parameters (in millions)
    Logger(f'LLM总参数量：{count_parameters(model) / 1e6:.3f} 百万')  # ("Total LLM parameters: ... million")
    return model, tokenizer


if __name__ == "__main__":
    # ==================== Command-line argument parsing ====================
    parser = argparse.ArgumentParser(description="Tiny-LLM Pretraining")
    
    # Basic training parameters
    parser.add_argument("--out_dir", type=str, default="base_model_215M", help="Output directory for the model")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Training device")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type")
    
    # Experiment tracking and data-loading parameters
    parser.add_argument("--use_swanlab", action="store_true", help="Whether to use SwanLab for experiment tracking")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of data-loading worker processes")
    parser.add_argument("--data_path", type=str, default="./seq_monkey_datawhale.jsonl", help="Path to the training data")
    
    # Training optimization parameters
    parser.add_argument("--accumulation_steps", type=int, default=8, help="Number of gradient accumulation steps")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping threshold")
    parser.add_argument("--warmup_iters", type=int, default=0, help="Number of learning-rate warmup iterations")
    
    # Logging and saving parameters
    parser.add_argument("--log_interval", type=int, default=100, help="Logging interval")
    parser.add_argument("--save_interval", type=int, default=1000, help="Model saving interval")
    
    # Multi-GPU training parameters
    parser.add_argument("--gpus", type=str, default='0,1,2,3,4,5,6,7', help="GPU IDs to use, separated by commas (e.g.: '0,1,2')")

    args = parser.parse_args()

    # ==================== GPU environment setup ====================
    # Set the visible GPU devices
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        # Automatically set the primary device to the first available GPU
        if torch.cuda.is_available():
            args.device = "cuda:0"
        else:
            args.device = "cpu"

    # ==================== Experiment tracking initialization ====================
    if args.use_swanlab:
        # Note: you need to log in first via swanlab.login(api_key='your key')
        run = swanlab.init(
            project="Happy-LLM",  # Project name
            experiment_name="Pretrain-215M",  # Experiment name
            config=args,  # Save all hyperparameters
        )

    # ==================== Model configuration ====================
    # Define the language model's configuration parameters
    lm_config = ModelConfig(
        dim=1024,      # Model dimension
        n_layers=18,   # Number of Transformer layers
    )

    # ==================== Training environment setup ====================
    max_seq_len = lm_config.max_seq_len  # Maximum sequence length
    args.save_dir = os.path.join(args.out_dir)  # Model save directory
    
    # Create the necessary directories
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Set the random seed to ensure reproducible results
    torch.manual_seed(42)
    
    # Determine the device type (for choosing the appropriate context manager)
    device_type = "cuda" if "cuda" in args.device else "cpu"

    # Set up the context manager for mixed-precision training
    # Use nullcontext for CPU training and autocast for GPU training
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

    # ==================== Model and data initialization ====================
    # Initialize the model and tokenizer
    model, tokenizer = init_model()
    
    # Create the training dataset
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=max_seq_len)
    
    # Create the data loader
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,  # Batch size
        pin_memory=True,             # Load data into pinned memory to speed up GPU transfer
        drop_last=False,             # Do not drop the last incomplete batch
        shuffle=True,                # Shuffle the data
        num_workers=args.num_workers # Number of parallel data-loading workers
    )

    # ==================== Optimizer and training component initialization ====================
    # Initialize the gradient scaler for mixed-precision training
    # Enable it only when using float16 or bfloat16
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    
    # Initialize the Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # ==================== Start training ====================
    # Compute the number of iterations per epoch
    iter_per_epoch = len(train_loader)
    
    # Start the training loop
    for epoch in range(args.epochs):
        train_epoch(epoch)
```
### 5.3.5 SFT Training

The SFT training code is basically the same as the pretraining code; only the imported Dataset differs. Here we use the SFTDataset for multi-turn dialogue training.

```python
import os
import platform
import argparse
import time
import warnings
import math
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader
from contextlib import nullcontext

from transformers import AutoTokenizer

from k_model import ModelConfig, Transformer
from dataset import SFTDataset

import swanlab

# Ignore warnings
warnings.filterwarnings('ignore')


def Logger(content):
    """Logger"""
    print(content)

def get_lr(it, all):
    """Get the learning rate"""
    # 1) linear warmup for warmup_iters steps
    # 1) linear warmup for the warmup iterations
    warmup_iters = args.warmup_iters
    lr_decay_iters = all
    min_lr = args.learning_rate / 10

    if it < warmup_iters:
        return args.learning_rate * it / warmup_iters
    
    # 2) if it > lr_decay_iters, return min learning rate
    # 2) if the iteration count exceeds the learning-rate decay iterations, return the minimum learning rate
    if it > lr_decay_iters:
        return min_lr
    
    # 3) in between, use cosine decay down to min learning rate
    # 3) in between, use cosine decay down to the minimum learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (args.learning_rate - min_lr)

def train_epoch(epoch):
    """Train one epoch"""
    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        # Get the learning rate and update the optimizer
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Forward pass
        with ctx:
            out = model(X, Y)
            loss = out.last_loss / args.accumulation_steps
            loss_mask = loss_mask.view(-1)
            loss = torch.sum(loss * loss_mask) / loss_mask.sum()

        # Backpropagation
        scaler.scale(loss).backward()

        # Update the weights
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        # Print the log
        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.7f} epoch_Time:{}min:'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))
            if args.use_swanlab:
                swanlab.log({
                    "loss": loss.item() * args.accumulation_steps,
                    "lr": optimizer.param_groups[-1]['lr']
                })

        # Save the model
        if (step + 1) % args.save_interval == 0:
            model.eval()
            ckp = f'{args.save_dir}/sft_dim{lm_config.dim}_layers{lm_config.n_layers}_vocab_size{lm_config.vocab_size}.pth'

            # Handle multi-GPU saving
            state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            torch.save(state_dict, ckp)
            model.train()
        
        # Save the model periodically
        if (step + 1) % 20000 == 0:
            model.eval()
            ckp = f'{args.save_dir}/sft_dim{lm_config.dim}_layers{lm_config.n_layers}_vocab_size{lm_config.vocab_size}_step{step+1}.pth'

            state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            torch.save(state_dict, ckp)
            model.train()


def init_model():
    """Initialize the model"""
    def count_parameters(model):
        """Count the model's parameters"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained('./tokenizer_k/')

    # Initialize the model
    model = Transformer(lm_config)

    # Load the pretrained weights
    ckp = './base_model_215M/pretrain_1024_18_6144.pth'
    state_dict = torch.load(ckp, map_location=args.device)
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
    
    # Multi-GPU initialization
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        Logger(f"Using {num_gpus} GPUs with DataParallel!")
        model = torch.nn.DataParallel(model)
    
    model = model.to(args.device)
    Logger(f'LLM总参数量：{count_parameters(model) / 1e6:.3f} 百万')  # ("Total LLM parameters: ... million")
    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tiny-LLM Pretraining")
    parser.add_argument("--out_dir", type=str, default="sft_model_215M", help="Output directory")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type")
    parser.add_argument("--use_swanlab", action="store_true", help="Whether to use SwanLab for experiment tracking")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of data-loading worker processes")
    parser.add_argument("--data_path", type=str, default="./BelleGroup_sft.jsonl", help="Path to the training data")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="Number of gradient accumulation steps")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping threshold")
    parser.add_argument("--warmup_iters", type=int, default=0, help="Number of warmup iterations")
    parser.add_argument("--log_interval", type=int, default=100, help="Logging interval")
    parser.add_argument("--save_interval", type=int, default=1000, help="Model saving interval")
    # Add the multi-GPU parameter
    parser.add_argument("--gpus", type=str, default='0,1,2,3,4,5,6,7', help="GPU IDs separated by commas (e.g. '0,1,2')")

    args = parser.parse_args()

    # Set the visible GPUs
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        # Automatically set the primary device to the first GPU
        if torch.cuda.is_available():
            args.device = "cuda:0"
        else:
            args.device = "cpu"

    # Initialize swanlab
    if args.use_swanlab:
        run = swanlab.init(
            project="Happy-LLM",
            experiment_name="SFT-215M",
            config=args,
        )

    # Model configuration
    lm_config = ModelConfig(
        dim=1024,
        n_layers=18,
    )
    max_seq_len = lm_config.max_seq_len
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(42)
    device_type = "cuda" if "cuda" in args.device else "cpu"

    # Context manager
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

    # Initialize the model and tokenizer
    model, tokenizer = init_model()
    
    # Create the dataset and data loader
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=max_seq_len)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=True,
        num_workers=args.num_workers
    )

    # Scaler and optimizer
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Start training
    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_epoch(epoch)
```


### 5.3.6 Generating Text with the Model

After the model training is complete, a model file will be generated under the `output` directory; this file is our trained model. We can use the following command to generate text.

```bash
python model_sample.py
```

Let's look at the code in the `model_sample.py` file; this file defines a `TextGenerator` class for generating text.

```python
import os
import pickle
from contextlib import nullcontext
import torch
from k_model import ModelConfig, Transformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

class TextGenerator:
    def __init__(self, 
                 checkpoint='./base_model_215M/pretrain_1024_18_6144.pth',  # Model checkpoint path
                 tokenizer_model_path='./tokenizer_k/',  # Tokenizer model path
                 seed=42,  # Random seed, ensures reproducibility
                 device=None,  # Device; prefer CUDA, fall back to CPU if no CUDA is available
                 dtype="bfloat16"):  # Data type, defaults to float32; can choose float16 or bfloat16
        """
        Initializes the TextGenerator class, loading the model, setting the device and tokenizer, etc.
        """
        # Model loading configuration
        self.checkpoint = checkpoint  # Saved model checkpoint path
        self.tokenizer_model_path = tokenizer_model_path  # Tokenizer model file path
        self.seed = seed  # Random seed, used for reproducible generation
        self.device = device or ('cuda:0' if torch.cuda.is_available() else 'cpu')  # Choose the device based on the hardware
        self.dtype = dtype  # The model's floating-point type
        self.device_type = 'cuda' if 'cuda' in self.device else 'cpu'  # Determine whether the current device is CUDA
        
        # Set random seeds to ensure reproducible generation
        torch.manual_seed(seed)  # Set the CPU random seed
        torch.cuda.manual_seed(seed)  # Set the CUDA random seed
        torch.backends.cuda.matmul.allow_tf32 = True  # Allow CUDA to use TF32 precision for matrix multiplication
        torch.backends.cudnn.allow_tf32 = True  # Allow cuDNN to use TF32 precision for acceleration
        
        # Choose the appropriate automatic mixed-precision context based on the dtype
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.dtype]
        self.ctx = nullcontext() if self.device_type == 'cpu' else torch.amp.autocast(device_type=self.device_type, dtype=ptdtype)
        
        # Load the model checkpoint file
        checkpoint_dict = torch.load(self.checkpoint, map_location=self.device)  # Load the model parameters # Initialize model parameters
        self.model = Transformer(ModelConfig(dim=1024, n_layers=18))  # Instantiate the Transformer model
        sunwanted_prefix = '_orig_mod.'
        for k, v in list(checkpoint_dict.items()):
            if k.startswith(sunwanted_prefix):
                checkpoint_dict[k[len(sunwanted_prefix):]] = checkpoint_dict.pop(k)
        self.model.load_state_dict(checkpoint_dict, strict=False)
        
        # Count the model's parameters
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model has {num_params / 1e6:.3f} M parameters.")
        # Set the model to evaluation mode, preventing training-mode operations such as dropout from affecting the results
        self.model.eval()
        # Place the model on the correct device (GPU or CPU)
        self.model.to(self.device)
        # Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_model_path)  # Load the tokenizer from the specified path

    def chat_template(self, prompt):
        message = [
            {"role": "system", "content": "你是一个AI助手，你的名字叫小明。"},  # ("You are an AI assistant, and your name is Xiaoming.")
            {"role": "user", "content": prompt}
        ]
        return self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

    def sft_sample(self, 
               start="Hello!",  # The starting prompt for the generated text; can be any string
               num_samples=3,  # Number of samples to generate, 3 by default
               max_new_tokens=256,  # Maximum number of tokens generated per sample, up to 256 by default
               temperature=0.7,  # Controls generation randomness; 1.0 is standard, larger values are more random
               top_k=300):  # Keep the top_k tokens with the highest probability, limiting the choice range during generation
        """
        Generates samples from the given starting text.
        
        :param start: the starting prompt for the generated text
        :param num_samples: the number of text samples to generate
        :param max_new_tokens: the maximum number of tokens generated per sample
        :param temperature: controls the randomness of generation; lower values are more deterministic, higher values more random
        :param top_k: restricts the range of tokens selected during generation
        :return: the list of generated text samples
        """
        start = self.chat_template(start)
        # Encode the starting text into a sequence of token ids
        start_ids = self.tokenizer(start).data['input_ids']
        # print('start_ids:', start_ids)
        x = (torch.tensor(start_ids, dtype=torch.long, device=self.device)[None, ...])  # Convert the encoded token ids into a PyTorch tensor
        generated_texts = []  # Holds the generated text samples
        with torch.no_grad():  # Disable gradient computation for efficiency
            with self.ctx:  # Enter the automatic mixed-precision context (when on a GPU using float16)
                for k in range(num_samples):  # Loop to generate the specified number of samples
                    y = self.model.generate(x, self.tokenizer.eos_token_id, max_new_tokens, temperature=temperature, top_k=top_k)  # Generate text
                    generated_texts.append(self.tokenizer.decode(y[0].tolist()))  # Decode the generated token sequence into readable text
        return generated_texts  # Return the generated text samples


    def pretrain_sample(self, 
               start="Hello!",  # The starting prompt for the generated text; can be any string
               num_samples=3,  # Number of samples to generate, 3 by default
               max_new_tokens=256,  # Maximum number of tokens generated per sample, up to 256 by default
               temperature=0.7,  # Controls generation randomness; 1.0 is standard, larger values are more random
               top_k=300):  # Keep the top_k tokens with the highest probability, limiting the choice range during generation
        """
        Generates samples from the given starting text.
        
        :param start: the starting prompt for the generated text
        :param num_samples: the number of text samples to generate
        :param max_new_tokens: the maximum number of tokens generated per sample
        :param temperature: controls the randomness of generation; lower values are more deterministic, higher values more random
        :param top_k: restricts the range of tokens selected during generation
        :return: the list of generated text samples
        """
        # If start starts with 'FILE:', the starting text is read from a file
        if start.startswith('FILE:'):
            with open(start[5:], 'r', encoding='utf-8') as f:
                start = f.read()  # Read the file contents as the starting text
        
        # Encode the starting text into a sequence of token ids
        start_ids = self.tokenizer(start).data['input_ids']
        # print('start_ids:', start_ids)
        x = (torch.tensor(start_ids, dtype=torch.long, device=self.device)[None, ...])  # Convert the encoded token ids into a PyTorch tensor
        # print(x.shape)
        generated_texts = []  # Holds the generated text samples
        with torch.no_grad():  # Disable gradient computation for efficiency
            with self.ctx:  # Enter the automatic mixed-precision context (when on a GPU using float16)
                for k in range(num_samples):  # Loop to generate the specified number of samples
                    y = self.model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)  # Generate text
                    generated_texts.append(self.tokenizer.decode(y[0].tolist()))  # Decode the generated token sequence into readable text
        
        return generated_texts  # Return the generated text samples
    
if __name__ == "__main__":
    print("------------------- Pretrain Sample ------------------- \n")

    pretrain_prompt_datas = [
        '<|im_start|>北京大学是',  # ('<|im_start|>Peking University is')
        '<|im_start|>中国矿业大学（北京）地球科学与测绘工程学院',  # ('<|im_start|>China University of Mining and Technology (Beijing), School of Geosciences and Surveying Engineering')
    ]

    generator = TextGenerator(checkpoint='./base_model_215M/pretrain_1024_18_6144.pth')  # Initialize the generator
    for i in range(len(pretrain_prompt_datas)):
        samples = generator.pretrain_sample(start=pretrain_prompt_datas[i], num_samples=1, max_new_tokens=120, temperature=0.75)
        print(f"\nSample {i+1}:\n{pretrain_prompt_datas[i]}{samples[0]}\n{'-'*20}")  # Print the generated sample, separated by a divider line

    print("\n ------------------- SFT Sample ------------------- \n")

    sft_prompt_datas = [
        '你好呀',  # ('Hello there')
        "中国的首都是哪里？",  # ("What is the capital of China?")
        "1+12等于多少？",  # ("What is 1+12?")
        "你是谁？"  # ("Who are you?")
    ]
    generator = TextGenerator(checkpoint='./sft_model_215M/sft_dim1024_layers18_vocab_size6144.pth')  # Initialize the generator
    for i in range(len(sft_prompt_datas)):
        samples = generator.sft_sample(start=sft_prompt_datas[i], num_samples=1, max_new_tokens=128, temperature=0.6)
        print(f"\nSample {i+1}:\nQuestion: {sft_prompt_datas[i]} \nAI answer: {samples[0]}\n{'-'*20}")  # Print the generated sample, separated by a divider line

```

Finally, let's look at the model's output results:

```
------------------- SFT Sample ------------------- 

Model has 215.127 M parameters.

Sample 1:
Question: 你好呀  (Hello there)
AI answer: 你好!有什么我可以帮你的吗?  (Hello! Is there anything I can help you with?)
--------------------

Sample 2:
Question: 中国的首都是哪里？  (What is the capital of China?)
AI answer: 中国的首都是北京。  (The capital of China is Beijing.)
--------------------

Sample 3:
Question: 1+1等于多少？  (What is 1+1?)
AI answer: 1+1等于2。  (1+1 equals 2.)
--------------------
------------------- Pretrain Sample ------------------- 

Model has 215.127 M parameters.

Sample 1:
<|im_start|>北京大学是中国最早建立的研究型大学之一,是我国最早设置研究生院的高校之一,是第一、二国教育委员会师资培训基地;北京大学是第一、二所国立大学,其校名与北京大学相同。  (Peking University is one of the earliest research universities established in China, one of the first universities to set up a graduate school, and a teacher-training base of the First and Second National Education Commissions; Peking University is among the first and second national universities, and its name is identical to Peking University.)
北京大学录取标准:本科三批1万元,本科一批1万元,本科一批2000元,专科一批2000元,高中起点:非本科一批  (Peking University admission standards: third-tier undergraduate 10,000 yuan, first-tier undergraduate 10,000 yuan, first-tier undergraduate 2,000 yuan, junior-college first tier 2,000 yuan, high-school starting point: not first-tier undergraduate)
--------------------

Sample 2:
<|im_start|>中国矿业大学（北京）地球科学与测绘工程学院副教授黄河流域地质学科带头人古建平教授为大家介绍世界地质变化的概念及工作经验。  (Gujianping, associate professor at the School of Geosciences and Surveying Engineering of China University of Mining and Technology (Beijing) and leader of the Yellow River basin geology discipline, introduces the concept of global geological change and his work experience.)
古建平教授介绍了最近几年的植物学和地质学的基本概念,尤其是树都黄河、松涛、暗河等都有地质学工作者的身影,其中树都黄河以分布面积最大,是树都黄河中华砂岩公园的主景区。  (Professor Gujianping introduced the basic concepts of botany and geology from recent years; in particular, geological workers can be found at Shudu Yellow River, Songtao, the Dark River and so on, among which Shudu Yellow River has the largest distribution area and is the main scenic area of the Chinese sandstone park at Shudu Yellow River.)
黄河内蒙古  (Yellow River, Inner Mongolia)
--------------------
```

At this point, our model has finished training. Congratulations on training your very own large model.

> When training, you can set the batch lower to reduce GPU memory usage and avoid running out of VRAM. Of course, this increases the training time; you can adjust the batch size according to your GPU's memory. In actual testing, with a pretraining batch of 4, only 7 GB of VRAM is needed, but the training time is expected to be 533 hours. The author trained on 8x 4090 GPUs: pretraining took 46 hours in total, and the SFT stage took 24 hours on BelleGroup's 3.5 million Chinese instructions.

The author has also uploaded the models trained in this chapter to the ModelScope platform. If your hardware is not sufficient to train a large model, you can download the models from ModelScope to debug and try them out. The model download links are as follows:

> *ModelScope model download link: [🤖 ModelScope](https://www.modelscope.cn/collections/Happy-LLM-e98b91b10b684a)*    
> *ModelScope Studio experience link: [🤖 Studio](https://www.modelscope.cn/studios/kmno4zx/happy_llm_215M_sft)*


**References**

[1] Andrej Karpathy. (2023). *llama2.c: Fullstack Llama 2 LLM solution in pure C*. GitHub repository. https://github.com/karpathy/llama2.c  

[2] Andrej Karpathy. (2023). *llm.c: GPT-2/GPT-3 pretraining in C/CUDA*. GitHub repository. https://github.com/karpathy/llm.c  

[3] Hugging Face. (2023). *Tokenizers documentation*. https://huggingface.co/docs/tokenizers/index  

[4] Skywork Team. (2023). *SkyPile-150B: A large-scale bilingual dataset*. Hugging Face dataset. https://huggingface.co/datasets/Skywork/SkyPile-150B  

[5] BelleGroup. (2022). *train_3.5M_CN: Chinese dialogue dataset*. Hugging Face dataset. https://huggingface.co/datasets/BelleGroup/train_3.5M_CN  

[6] Jingyao Gong. (2023). *minimind: Minimalist LLM implementation*. GitHub repository. https://github.com/jingyaogong/minimind  

[7] Mobvoi. (2023). *seq-monkey-data: Llama2 training/inference data*. GitHub repository. https://github.com/mobvoi/seq-monkey-data
