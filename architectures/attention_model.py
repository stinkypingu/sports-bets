import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):
    def __init__(self, E, head_size, dropout):
        super().__init__()

        self.head_size = head_size

        self.key = nn.Linear(E, head_size, bias=False)
        self.query = nn.Linear(E, head_size, bias=False)
        self.value = nn.Linear(E, head_size, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #B,P,E = x.shape
        k = self.key(x)     #(B,P,E) @ (E,head_size) -> (B,P,head_size)
        q = self.query(x)   #(B,P,E) @ (E,head_size) -> (B,P,head_size)

        wei = q @ k.transpose(-1,-1) * (self.head_size ** -0.5)  #(B,P,head_size) @ (B,head_size,P) -> (B,P,P) scaled dot product attention
        wei = F.softmax(wei, dim=-1)    #(B,P,P) weighted
        wei = self.dropout(wei)

        v = self.value(x)   #(B,P,E) @ (E,head_size) -> (B,P,head_size)
        out = wei @ v       #(B,P,P) @ (B,P,head_size) -> (B,P,head_size)

        return out
    

class MultiHeadAttention(nn.Module):
    def __init__(self, E, head_size, num_heads, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(E=E, head_size=head_size, dropout=dropout) for _ in range(num_heads)])

        self.proj = nn.Linear(head_size * num_heads, E)     #(E,E) in this case but in other cases could be (x,E)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)     #(B,P,head_size) * (_,_,num_heads) -> (B,P,head_size * num_heads)
        out = self.proj(out)                                    #(B,P,head_size * num_heads) @ (head_size * num_heads,E) -> (B,P,E)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, E, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(E, 4 * E),
            nn.ReLU(),
            nn.Linear(4 * E, E),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
    

class Block(nn.Module):
    def __init__(self, E, num_heads, dropout):
        super().__init__()

        head_size = E // num_heads

        self.ln1 = nn.LayerNorm(E)
        self.sa = MultiHeadAttention(E=E, head_size=head_size, num_heads=num_heads, dropout=dropout)

        self.ln2 = nn.LayerNorm(E)
        self.ff = FeedForward(E=E, dropout=dropout)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x
    

class GPT(nn.Module):
    def __init__(self, E, num_heads, num_layers, num_targets, dropout):
        super().__init__()

        self.blocks = nn.Sequential(*[Block(E=E, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)])

        self.lnf = nn.LayerNorm(E)              #final layernorm
        self.fff = nn.Linear(E, num_targets)    #final linear, output targets

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x, targets=None):
        B,P,E = x.shape

        x = self.blocks(x)                      #(B,P,E) -> (B,P,E)
        x = self.lnf(x)                         #(B,P,E) -> (B,P,E)
        predictions = self.fff(x)               #(B,P,E) -> (B,P,1)

        if targets is None:
            loss = None

        else:
            predictions = predictions.view(B * P)   #(B * P,)
            targets = targets.view(B * P)           #(B * P,)
            loss = F.smooth_l1_loss(predictions, targets)
        
        return predictions, loss

