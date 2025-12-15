import math
from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn as nn
from torch.nn import functional as F
from Trainer import Trainer
import matplotlib.pyplot as plt
from time import sleep
import scienceplots

plt.style.use(["science", "grid"])

#import tiktoken

# -----------------------
# Data utilities
# -----------------------

class Tokenizer:
    def __init__(self, text: str):
        self.chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, s: str):
        return [self.stoi[c] for c in s]

    def decode(self, l):
        return "".join([self.itos[i] for i in l])


class DataLoader:
    """
    Gestiona los splits de entrenamiento/validación y provee lotes
    de secuencias de caracteres tokenizados.
    """

    def __init__(self, data, train_size: float = 0.9, block_size: int = 32, batch_size: int = 64):
        data = torch.tensor(data, dtype=torch.long)
        n = int(train_size * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]
        self.block_size = block_size
        self.batch_size = batch_size

    def get_batch(self, split):
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i : i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + self.block_size + 1] for i in ix])
        return x, y


# -----------------------
# Model components
# -----------------------


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (B, T, C)
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return x


def attention(query, key, value, mask: bool = False, dropout_layer=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask:
        _, T, _ = query.shape
        tril = torch.tril(torch.ones(T, T, device=query.device))
        scores = scores.masked_fill(tril == 0, float("-inf"))
    scores = F.softmax(scores, dim=-1)
    if dropout_layer is not None:
        scores = dropout_layer(scores)
    return torch.matmul(scores, value)


class Head(nn.Module):
    """One head of self-attention."""

    def __init__(self, n_embd, head_size, dropout: float = 0.0):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask: bool = False):
        key = self.key(x)
        query = self.query(x)
        value = self.value(x)
        return attention(query, key, value, mask, dropout_layer=self.dropout)


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel."""

    def __init__(self, n_embd, num_heads, dropout: float = 0.0):
        super().__init__()
        assert n_embd % num_heads == 0, "n_embd must be divisible by num_heads"
        head_size = n_embd // num_heads
        self.heads = nn.ModuleList([Head(n_embd, head_size, dropout=dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask: bool = False):
        out = torch.cat([h(x, mask=mask) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity."""

    def __init__(self, n_embd, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation."""

    def __init__(self, n_embd, n_heads, mask: bool = False, dropout: float = 0.0):
        super().__init__()
        self.mask = mask
        self.msa = MultiHeadAttention(n_embd, n_heads, dropout=dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd, dropout=dropout)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.msa(self.ln1(x), mask=self.mask)
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, dropout: float = 0.0):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = PositionalEncoding(n_embd)
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, mask=True, dropout=dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_embd = self.token_embedding_table(idx)  # (B, T, C)
        x = self.position_embedding_table(tok_embd)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, block_size=32):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


@dataclass
class GPTConfig:
    vocab_size: int
    n_embd: int = 384
    n_head: int = 6
    n_layer: int = 6
    dropout: float = 0.2


# -----------------------
# Training / evaluation
# -----------------------


def main():
    torch.manual_seed(2025)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Hyperparameters
    learning_rate = 3e-4
    max_iters = 5000
    eval_interval = 100
    eval_iters = 200
    dropout = 0.2
    block_size = 128
    batch_size = 32

    # Load data
    text_path = Path("input.txt")
    if not text_path.exists():
        raise FileNotFoundError("input.txt not found. Download it first.")
    text = text_path.read_text(encoding="utf-8")

    tokenizer = Tokenizer(text)
    #tokenizer = tiktoken.get_encoding("gpt2")
    data_loader = DataLoader(tokenizer.encode(text), batch_size=batch_size, block_size=block_size)

    config = GPTConfig(vocab_size=len(tokenizer.chars), dropout=dropout)
    model = GPTLanguageModel(
        config.vocab_size, config.n_embd, config.n_head, config.n_layer, dropout=config.dropout
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        data_loader=data_loader,
        device=device,
        max_iters=max_iters,
        eval_interval=eval_interval,
        eval_iters=eval_iters,
    )
    gpu = f"GPU: {torch.cuda.get_device_name(device)}" if torch.cuda.is_available() else "CPU"
    print(f"Starting training on {gpu}...")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")
    trainer.fit()
    print(f"Training completed in {trainer.training_time / 60:.2f} minutes")
    trainer.plot_learning_curve(title="Tiny Shakespeare - Learning curve")
    torch.save(model.state_dict(), "model.pkl")

    # Generation example
    max_new_tokens = 1000
    context = torch.zeros((1, 1), dtype=torch.long, device=device)

    with torch.no_grad():
        try:
            while True: # Infinite loop
                context = model.generate(context, max_new_tokens=1, block_size=block_size) # Generate a new token
                new_token = context[0, -1].item() # Get the last token generated
                print(tokenizer.decode([new_token]), end="", flush=True) # Print the new token decoded
                context = context[:, -block_size:] # Update the context to the last block_size tokens
                sleep(0.03) # Wait 30 milliseconds before generating the next token
        except KeyboardInterrupt:
            print("\nGeneration stopped by user.")


if __name__ == "__main__":
    main()
