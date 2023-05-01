import torch
from torch import nn
import torch.nn.functional as F
from math import sqrt

n_embed = 64
T = 8
num_heads = 4
num_layers = 4

# Head
#   q = linear1(x)
#   k = linear2(x)
#   v = linear3(x)
#   weights = q @ k^T
#   masking stuff

# Multi-head attention
#   for i in num_heads:
#     do Head // num_heads
#     concat output
#     linear layer on output

class GPT(nn.Module):
  def __init__(self):
    super().__init__()
    self.layers = [MultiHeadAttentionBlock() for _ in range(4)]

  def forward(x):
    x = x + self.head


class MultiHeadAttentionBlock(nn.Module):
  def __init__():
    super().__init__()
  
  def forward(x):
    x 


# single head attention
# 1. takes T * n_embed
# 2. performs self attention
class Head(nn.Module):
  def __init__(self, head_size) -> None:
    super().__init__()
    self.key = nn.Linear(n_embed, head_size, bias=False)
    self.query = nn.Linear(n_embed, head_size, bias=False)
    self.value = nn.Linear(n_embed, head_size, bias=False)
    self.tril = self.register_buffer('tril')
  
  def forward(self, x):
    # B,T,C = x.shape
    k = self.key(x)   # (B,T,head_size) (?)
    q = self.query(x) # (B,T,head_size) (?)
    att_wei = q @ k.transpose(-1, -2) / k.shape[-1] ** -0.5 # (B, T, T)
    att_wei = mask it
    att_wei = F.softmax(att_wei, dim=-1) # (B,T,T)
    
    v = self.value(x) # (B,T,head_size)
    output = att_wei @ v



Head()