import torch
import numpy as np
import torch.nn as nn



class self_attention(nn.Module):

    def __init__(self, d_in , d_out , context_length , drop_out , num_heads , qkv_bias= False ):
        super().__init__()

        self.d_out = d_out
        self.heads = num_heads
        self.head_dim = d_out // num_heads
        self.query = torch.nn.Linear(d_in, d_out , bias=qkv_bias)
        self.key = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.value = torch.nn.Linear(d_in , d_out, bias=qkv_bias)
        self.dropout = torch.nn.Dropout(drop_out)
        self.out_proj = torch.nn.Linear(d_out , d_out)
        self.sofmax = torch.nn.Softmax(dim = -1)
        self.register_buffer( 'mask' , torch.triu(torch.ones(context_length , context_length) , diagonal= 1))


    def forward(self, x):

        b, num_tokens , d_in  = x.size()

        w_q = self.query(x)
        w_k = self.key(x)
        w_v = self.value(x)

        w_k = w_k.view(b,num_tokens,self.heads,self.head_dim)
        w_q = w_q.view(b,num_tokens,self.heads,self.head_dim)
        w_v = w_v.view(b,num_tokens,self.heads,self.head_dim)

        w_k = w_k.transpose(1,2)
        w_q = w_q.transpose(1,2)
        w_v = w_v.transpose(1,2)

        attention = w_q @ w_k.transpose(2,3)
        attention.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens] , -torch.inf)
        attention = self.sofmax(attention/ (w_k.shape[-1])**0.5)
        attention = self.dropout(attention)
        content_vec = (attention @ w_v).transpose(1,2)
        content_vec = content_vec.contiguous().view(b,num_tokens,self.d_out)
        content_vec = self.out_proj(content_vec)
        return content_vec



"""
class multiheadAttentionWraper(nn.Module):

    def __init__(self, d_in , d_out , context_length , drop_out , num_heads, qkv_bias= False):
        super.__init__()
        self.heads = nn.ModuleList([ self_attention(d_in , d_out , context_length , drop_out,qkv_bias) for _ in range(num_heads)])
    
    def forward(self ,x):
        return torch.cat([head(x) for head in self.heads], dim = -1)

"""