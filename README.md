# Small Transformer Language Model

A small **decoder-only Transformer language model implemented from scratch using PyTorch**.

This project was built as a learning and experimentation project to understand the internal components of modern Transformer-based language models, including token embeddings, positional embeddings, causal self-attention, multi-head attention, feed-forward networks, residual connections, and language-model training.

> **Note:** This is an educational project rather than a production-ready conversational AI system. The main goal is to demonstrate an understanding of the architecture and training process.

## Architecture

The model follows a decoder-only Transformer architecture:

```text
Input Tokens
     │
     ▼
Token Embedding
     +
Positional Embedding
     │
     ▼
┌──────────────────────┐
│   Transformer Block  │
│                      │
│  LayerNorm           │
│      ↓               │
│  Multi-Head          │
│  Self-Attention      │
│      ↓               │
│  Residual Connection │
│      ↓               │
│  LayerNorm           │
│      ↓               │
│  Feed-Forward Network│
│      ↓               │
│  Residual Connection │
└──────────────────────┘
     │
     │ × 4
     ▼
Linear Projection
     │
     ▼
Token Probabilities
```

The Transformer blocks use **pre-layer normalization**, residual connections, multi-head causal self-attention, and feed-forward networks.

## Model Configuration

| Parameter            |     Value |
| -------------------- | --------: |
| Transformer blocks   |         4 |
| Embedding dimension  |        96 |
| Attention heads      |        12 |
| Head dimension       |         8 |
| Context length       |       128 |
| Batch size           |        64 |
| Training steps       |     5,000 |
| Optimizer            |     AdamW |
| Learning rate        |      1e-3 |
| Trainable parameters | 5,754,802 |

## Dataset

The model is trained on the **DailyDialog** dataset.

The dataset is preprocessed into a dialogue-oriented text format and tokenized before being converted into PyTorch tensors.

The training data is split into:

* 80% training data
* 20% test data

Training examples are created using a next-token prediction objective.

For a sequence:

```text
The cat is
```

the model is trained to predict:

```text
cat is ...
```

More generally:

```text
X = [t₁, t₂, t₃, ..., tₙ]

Y = [t₂, t₃, t₄, ..., tₙ₊₁]
```

This allows the model to learn autoregressive language modelling.

## Tokenization

A pre-trained tokenizer is loaded using the Hugging Face `tokenizers` library.

The tokenizer converts the dialogue text into integer token IDs, which are then passed to an embedding layer.

```python
tokenizer = Tokenizer.from_file(tokenizer_path)

tokens = tokenizer.encode(text)

data = torch.tensor(
    tokens.ids,
    dtype=torch.long,
    device=device
)
```

## Causal Self-Attention

The attention mechanism is implemented manually rather than using a high-level Transformer module.

Each attention head creates:

* Query
* Key
* Value

and computes scaled dot-product attention:

```text
Attention(Q, K, V)
    = softmax(QKᵀ / √dₖ)V
```

A causal mask is applied so that a token cannot attend to future tokens.

This is important for autoregressive language modelling because, during training, the model should only use information that would have been available at generation time.

## Multi-Head Attention

Multiple attention heads are implemented using separate `Head` modules.

Their outputs are concatenated and projected back to the model embedding dimension.

```python
self.heads = nn.ModuleList([
    Head()
    for _ in range(n_head)
])

...

x = torch.cat(
    [head(x) for head in self.heads],
    dim=-1
)

return self.proj(x)
```

## Training

The model is trained using cross-entropy loss:

```python
logits = model(Xb)

loss = F.cross_entropy(
    logits.transpose(1, 2),
    Yb
)
```

The training loop includes:

* Forward pass
* Cross-entropy loss
* Backpropagation
* AdamW optimization
* Periodic train/test loss evaluation
* Checkpoint saving
* Basic overfitting detection

Checkpoints contain:

```python
{
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "loss": loss,
    "step": step
}
```

This allows training to be resumed from a previous checkpoint.

## Evaluation

The model is evaluated on a separate test dataset.

The recorded test loss in the notebook was approximately:

```text
4.1979
```

The project is primarily intended as an architectural and implementation exercise, so this metric should be interpreted in the context of the small model size and dataset rather than as a benchmark against large language models.

## Text Generation

After training, the model can generate text autoregressively.

A prompt is formatted using dialogue markers:

```text
<user> hello <assistant>
```

The model then repeatedly:

1. Runs the current sequence through the Transformer.
2. Takes the logits corresponding to the final token.
3. Converts logits to probabilities using softmax.
4. Samples the next token.
5. Appends the token to the sequence.

Example:

```python
generate("hello")
```

Example output from the trained model:

```text
<user> hello <assistant> I have to make some clothes to know .
You are your parents . <user>
```

The generated text is not consistently coherent, which is expected for a small experimental model trained on limited data.

## Project Structure

A typical project structure is:

```text
.
├── LLM.ipynb
├── dataset/
│   ├── dailyDialogCleand.txt
│   └── dailyDialogCleandTest.txt
├── tokenization/
│   └── dialogtokens.json
├── checkpoints/
│   └── ...
└── model1.pth
```

## What I Learned

This project was mainly an exercise in understanding how Transformer language models work internally.

The implementation helped me understand:

* Token embeddings
* Positional embeddings
* Query, Key and Value projections
* Scaled dot-product attention
* Causal masking
* Multi-head attention
* Feed-forward networks
* Layer normalization
* Residual connections
* Autoregressive language modelling
* Next-token prediction
* Cross-entropy loss
* Training and evaluation loops
* Model checkpointing
* Text generation

Rather than relying entirely on a pre-built Transformer implementation, the core architecture was implemented directly with PyTorch modules.

## Limitations

This project has several important limitations.

### Small model

The model contains approximately **5.75 million trainable parameters**, making it useful for experimentation but far too small to compete with modern large language models.

### Limited training data

The model is trained on the DailyDialog dataset, which is relatively small compared with the datasets used to train modern LLMs.

### Limited generation quality

The generated responses can be grammatically incorrect, incoherent, or unrelated to the input prompt.

### Experimental implementation

The project prioritizes understanding the architecture over production-level performance, efficiency, and generation quality.

## Future Improvements

Possible improvements include:

* Adding dropout
* Improving the training/validation pipeline
* Learning-rate scheduling
* Better checkpoint management
* Temperature-controlled sampling
* Top-k / top-p sampling
* Increasing model capacity
* Training on a larger dataset
* Improving data preprocessing
* Experimenting with different positional encoding methods
* Adding more Transformer blocks
* Comparing different model configurations

## Technologies

* Python
* PyTorch
* Hugging Face Tokenizers
* Jupyter Notebook

## Purpose

This repository is part of my practical exploration of **Transformer architectures and language modelling**.

The goal is not to build a state-of-the-art chatbot, but to demonstrate that I understand and can implement the fundamental components required to build a Transformer-based language model.
