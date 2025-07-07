import os
from self_attention_later import self_attention
from Data_preprocess import Dataload
from torch.utils.data import Dataset , DataLoader
from pathlib import Path
from tiktoken import get_encoding
from gptmodel import gptmodel
import torch
import json



text_file = os.path.join(Path(__file__).parent.resolve() , "the-verdict.txt")
traindata = Dataload( text_file , 50 , 3)
testdata = Dataload(text_file , 50, 3 , train= False)

trainloader = DataLoader( dataset=traindata, batch_size= 2 , shuffle= True )
testloader = DataLoader( dataset=testdata, batch_size= 2 , shuffle= True )

for lable , target in iter(trainloader):
    print(lable.shape, target.shape)
    print( lable , target)
    break
loss = torch.nn.CrossEntropyLoss()

def calc_loss( inputs , targets , model):
    preds = model(inputs)
    preds = preds.flatten(0,1)
    targets = targets.flatten(0)
    return loss(preds, targets)


def cal_loss_loader(  data_loader , model , device , num_batches):
    total_loss = 0
    i = 0
    for input_data , target_data in data_loader:
        total_loss+= calc_loss( input_data.to(device= device) , target_data.to(device=device) , model)
        i+=1
        if( i == num_batches):
            break

    return total_loss

def evaluate_model( model , train_loader , val_loader , device , eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = cal_loss_loader( train_loader , model ,device , num_batches = eval_iter) 
        val_loss = cal_loss_loader( val_loader , model ,device , num_batches = eval_iter) 
    model.train()
    return train_loss,  val_loss

def generate_and_print_sample( model , tokenizer , device , start_context):
    model.eval() 
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(model = model , batch = encoded , max_iter = 25, eos_id=tokenizer.encode('<|endoftext|>' , allowed_special = {'<|endoftext|>'})[0])
    decoded_text = token_ids_to_text( token_ids , tokenizer)
    print(decoded_text.replace( "\n", " "))
    model.train()


def generate_text_simple( model ,batch , max_iter, eos_id = None):
    num_sentence , _ = batch.shape
    for i in range(max_iter):
        
        output = model(batch)
        final_logits = torch.softmax(torch.stack([ logits[-1][:] for logits in output]), dim = -1)
        
        value , _ = torch.topk(final_logits , 4)
        min_value = value[:, -1]
        final_logits = torch.where( final_logits < min_value , torch.tensor(float(0)).to(final_logits.device) , final_logits)
        final_logits = final_logits / 1 #temprature scaling
        
        next_words = torch.multinomial(final_logits, num_samples=1) #torch.argmax(output , dim = -1)#.reshape(num_sentence , 1 )
        #print(next_words,final_logits[0][int(next_words[0][0])]) #[int(next_words[0][0])])
        if( eos_id and next_words == eos_id):
            break
        batch = torch.cat( (batch , next_words) , dim =1 )
        
    return batch


def token_ids_to_text( tokens , tokenizer):

    flat = tokens.squeeze(0).tolist()
    return tokenizer.decode(flat)

def text_to_token_ids( text , tokenizer ):
    
    tokens = tokenizer.encode(text , allowed_special = {'<|endoftext|>'})
    tokens = torch.tensor(tokens).unsqueeze(0)
    return tokens

def train_model_simple( model , train_loader , val_loader , optimizer , device , num_epochs , eval_freq , eval_iter , start_context , tokenizer):
    train_losses , val_losses , track_tokens_seen = [] , [] ,[]
    tokens_seen , global_step = 0 , -1 
    for epoch in range(num_epochs):
        model.train()
        for input_batch , target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss( input_batch.to(device= device), target_batch.to(device=device), model)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step+= 1
            print(f" this is the global step bfore eval condition : {global_step}")
            if( global_step % eval_freq == 0):
                train_loss , val_loss = evaluate_model( model , train_loader , val_loader , device , eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (step {global_step:06d}): "
                       f"Train loss {train_loss:.3f},"
                    f"Val loss {val_loss:.3f}")
        generate_and_print_sample(model , tokenizer, device, start_context)
    return train_losses, val_losses , track_tokens_seen
"""
config = "./ModelArchitecture.json"

with open(config , "r") as file:
    cfg = json.load(file)

model = gptmodel(cfg["GPT_CONFIG_124M"] , device="cuda")
optimizer = torch.optim.AdamW(model.parameters() , lr = 0.0004 , weight_decay=0.1 )
train_losses , val_losses  , toekns_seen = train_model_simple( model ,trainloader , testloader , optimizer , "cuda" , num_epochs=2 , eval_freq=5 , eval_iter=2 , start_context="Every effort moves you", tokenizer=get_encoding("gpt2") )
"""




"""

tokenizer = get_encoding("gpt2")
batch = []
input1 = "every effort moves"
input2 = "I really like"

target1 = " effort moves you"
target2 = " really like chocolate"



inputs = torch.cat((text_to_tokens(input1, tokenizer) , text_to_tokens(input2 , tokenizer)) , dim = 0)
targets = torch.cat((text_to_tokens(target1 , tokenizer) , text_to_tokens(target2, tokenizer)), dim = 0)




config = "./ModelArchitecture.json"

with open(config , "r") as file:
    cfg = json.load(file)


model = gptmodel(cfg["GPT_CONFIG_124M"])


preds = generate_output(model , inputs , 1)


preds = preds.flatten(0,1)
targets = targets.flatten(0)
print(preds, targets)
loss = torch.nn.CrossEntropyLoss()
print(torch.nn.functional.cross_entropy( preds, targets))

file.close()
"""