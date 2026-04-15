import torch
import torch.nn as nn
import math
class WorldModel(nn.Module):
    def __init__(self,vocab_size=11000,n_embd=256,n_head=8,n_layer=6,block_size=1024):
        super().__init__()
        self.token_emb=nn.Embedding(vocab_size,n_embd)
        self.pos_emb=nn.Parameter(torch.zeros(1,block_size,n_embd))
        encoder_layer=nn.TransformerEncoderLayer(
            d_model=n_embd,
            nhead=n_head,
            dim_feedforward=n_embd*4,
            batch_first=True
        )
        self.blocks=nn.TransformerEncoder(encoder_layer,num_layers=n_layer)
        self.ln_f=nn.LayerNorm(n_embd)
        self.head=nn.Linear(n_embd,vocab_size)
    def forward(self,idx):
        B,T=idx.shape
        mask=torch.triu(torch.ones(T,T),diagonal=1).bool().to(idx.device)
        x=self.token_emb(idx)+self.pos_emb[:,:T,:]
        x=self.blocks(x,mask=mask,is_causal=True)
        x=self.ln_f(x)
        logits=self.head(x)
        return logits
