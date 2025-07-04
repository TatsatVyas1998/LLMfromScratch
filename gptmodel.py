import torch
import torch.nn as nn
from self_attention_later import self_attention



class gptmodel(nn.Module):

    def __init__(self , cfg):

        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"] , cfg["embeding_dem"])
        self.pos_emb = nn.Embedding(cfg["context_length"] , cfg["embeding_dem"])   #each token will be size of embeding_dem that has values ranging from 0 to vocab_size. the context_length will be number of token at each context window.
        self.drop_emb = nn.Dropout(cfg["dropout_rate"])

        self.trf_blocks = nn.Sequential( *[Transformerblock(cfg) for _ in range(cfg["n_layers"])])
        
        self.final_norm = LayerNorm(cfg["embeding_dem"])

        self.out_head = nn.Linear( cfg["embeding_dem"] ,cfg["vocab_size"] , bias= False )




    
    def forward(self , int_idx):

        batch_size , seq_len = int_idx.shape
        tok_emb = self.tok_emb(int_idx) 
        pos_emb = self.pos_emb( torch.arange(seq_len))

        x = tok_emb + pos_emb

        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x= self.final_norm(x)
        x = self.out_head(x)

        return x

    

class Transformerblock(nn.Module):

    def __init__(self, cfg):
        
        super().__init__()

        self.att = self_attention(d_in= cfg["embeding_dem"] , d_out=cfg["embeding_dem"], context_length=cfg["context_length"] , drop_out=cfg["dropout_rate"] , num_heads=cfg["n_heads"]
                                , qkv_bias= cfg["qkv_bias"] )
        self.norm1 = LayerNorm(cfg["embeding_dem"])
        self.norm2= LayerNorm(cfg["embeding_dem"])
        self.ff= FeedForward(cfg)
        self.drop_shortcut = nn.Dropout(cfg["dropout_rate"])
        
    
    def forward(self , x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)

        x = x +shortcut
        shortcut= x

        x = self.norm2(x)
        x = self.ff(x)
        x= self.drop_shortcut(x)

        x = x+shortcut

        return x

class LayerNorm(nn.Module):

    def __init__(self, normalized_shape, eps=1e-5):

        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(normalized_shape))
        self.shift = nn.Parameter(torch.zeros(normalized_shape))
    
    def forward(self, x):
        mean = x.mean( dim = -1 , keepdim= True)
        var = x.var( dim = -1 , keepdim = True, unbiased = False )
        out = (x - mean)/ torch.sqrt(var + self.eps)
        return self.scale * out + self.shift
    

class FeedForward(nn.Module):

    def __init__(self , cfg):

        super().__init__()
        self.layer = nn.Sequential(nn.Linear(cfg["embeding_dem"], 4*cfg["embeding_dem"]) , nn.GELU() , nn.Linear(4*cfg["embeding_dem"] , cfg["embeding_dem"]))

    def forward(self,x):

        return self.layer(x)