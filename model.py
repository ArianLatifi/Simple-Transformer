import torch
import torch.nn as nn
import torch.nn.functional as F



class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd,4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 *n_embd,n_embd)
        )
    def forward(self,x):
        return self.net(x)

class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd,head_size,bias=False)
        self.query = nn.Linear(n_embd,head_size,bias=False)
        self.value = nn.Linear(n_embd,head_size,bias=False)
        self.register_buffer(
                    "tril",
                    torch.tril(
                        torch.ones(block_size, block_size)
                    )
                )
    def forward(self,x):
        B,T,C = x.shape

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        wei = q @ k.transpose(-2,-1)
        wei = wei * (k.shape[-1] ** -0.5)
        #Causal mask
        wei = wei.masked_fill(
            self.tril[:T,:T] == 0,
            float("-inf")
        )

        wei = F.softmax(
            wei,
            dim = -1
        )

        out = wei @ v

        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head,n_embd, head_size, block_size):
        super().__init__()
        self.heads = nn.ModuleList([
                Head(n_embd, head_size, block_size)
                for _ in range(n_head)
            ])
        self.proj = nn.Linear(n_embd,n_embd)
    def forward(self, x):
        x = torch.cat(
                [head(x) for head in self.heads],
                dim=-1
            )

        return self.proj(x)



class TransformerBlock(nn.Module):
    def __init__(self, n_embd, head_size, block_size):
        super().__init__()

        self.attention = MultiHeadAttention(n_head=12, n_embd=n_embd, head_size=head_size, block_size=block_size)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ffw = FeedForward(n_embd=n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self,x):
        x = x + self.attention(self.ln1(x))
        x = x + self.ffw(self.ln2(x))

        return x

class LanguageModel(nn.Module):

    def __init__(self, vocab_size, n_embd, block_size, device):
        super().__init__()
        self.device = device
        self.token_embbeding = nn.Embedding(vocab_size,n_embd,device=device) #(B,T,C)
        self.pos_embedding = nn.Embedding(block_size,n_embd,device=device)
        self.blocks = nn.Sequential(
                    *[
                        TransformerBlock(n_embd=n_embd, head_size=n_embd//12, block_size=block_size)
                        for _ in range(4)
                    ]
                )
        self.out = nn.Linear(n_embd ,vocab_size)

    def forward(self,x):
        B,T = x.shape
        t_embd = self.token_embbeding(x)
        p_embd = self.pos_embedding(torch.arange(T,device=self.device))
        x = t_embd + p_embd
        x = self.blocks(x)
        logits = self.out(x)
        return logits
