import torch
from gpt_download import download_and_load_gpt2
import numpy as np
from gptmodel import gptmodel
import json
from train import generate_and_print_sample
from tiktoken import get_encoding

def assign(left, right):
    if( left.shape != right.shape):
        raise ValueError(f"shape mismatch, Left: {left.shape}, and Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


def load_gpt_weights(model):
    
    settings , params = download_and_load_gpt2( model_size="124M" , models_dir = "gpt2")
    
    model.pos_emb.weight = assign( model.pos_emb.weight, params['wpe'])
    model.tok_emb.weight = assign( model.tok_emb.weight , params['wte'])
    for i in range( len( params["blocks"])):
        query_pre , key_pre , value_pre = np.split( (params["blocks"][i]["attn"]["c_attn"])["w"] , 3 , axis = -1 )
        model.trf_blocks[i].att.query.weight = assign(model.trf_blocks[i].att.query.weight , query_pre.T )
        model.trf_blocks[i].att.key.weight = assign(model.trf_blocks[i].att.key.weight , key_pre.T)
        model.trf_blocks[i].att.value.weight =  assign(model.trf_blocks[i].att.value.weight , value_pre.T)
        query_b_pre , key_b_pre , value_b_pre = np.split( (params["blocks"][i]["attn"]["c_attn"])["b"] , 3 , axis = -1 )
        model.trf_blocks[i].att.query.bias = assign(model.trf_blocks[i].att.query.bias , query_b_pre )
        model.trf_blocks[i].att.key.bias = assign(model.trf_blocks[i].att.key.bias , key_b_pre)
        model.trf_blocks[i].att.value.bias =  assign(model.trf_blocks[i].att.value.bias , value_b_pre)
        model.trf_blocks[i].att.out_proj.weight = assign ( model.trf_blocks[i].att.out_proj.weight , (params["blocks"][i]["attn"]["c_proj"])["w"].T)
        model.trf_blocks[i].att.out_proj.bias = assign ( model.trf_blocks[i].att.out_proj.bias , (params["blocks"][i]["attn"]["c_proj"])["b"])
        model.trf_blocks[i].ff.layer[0].weight = assign( model.trf_blocks[i].ff.layer[0].weight , params["blocks"][i]["mlp"]["c_fc"]["w"].T )
        model.trf_blocks[i].ff.layer[0].bias = assign( model.trf_blocks[i].ff.layer[0].bias , params["blocks"][i]["mlp"]["c_fc"]["b"] )
        model.trf_blocks[i].ff.layer[2].weight = assign( model.trf_blocks[i].ff.layer[2].weight , params["blocks"][i]["mlp"]["c_proj"]["w"].T )
        model.trf_blocks[i].ff.layer[2].bias = assign( model.trf_blocks[i].ff.layer[2].bias , params["blocks"][i]["mlp"]["c_proj"]["b"] )
        model.trf_blocks[i].norm1.scale = assign( model.trf_blocks[i].norm1.scale, params["blocks"][i]["ln_1"]["g"])
        model.trf_blocks[i].norm1.shift = assign( model.trf_blocks[i].norm1.shift, params["blocks"][i]["ln_1"]["b"])
        model.trf_blocks[i].norm2.scale = assign( model.trf_blocks[i].norm2.scale, params["blocks"][i]["ln_2"]["g"])
        model.trf_blocks[i].norm2.shift = assign( model.trf_blocks[i].norm2.shift, params["blocks"][i]["ln_2"]["b"])
    
    model.final_norm.scale = assign(model.final_norm.scale , params["g"])
    model.final_norm.shift = assign(model.final_norm.shift , params["b"])
    model.out_head.weight = assign( model.out_head.weight , params["wte"])
    


config = "./ModelArchitecture.json"

with open(config , "r") as file:
    cfg = json.load(file)

model = gptmodel(cfg["GPT_CONFIG_124M"] , device="cuda")

load_gpt_weights(model)
model.to(device='cuda')
generate_and_print_sample(model , tokenizer=get_encoding("gpt2") , device = "cuda" ,  start_context="Every effort moves you")