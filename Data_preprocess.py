import numpy as np
import torch
import torch.nn as nn
import re

from tiktoken import get_encoding

from torch.utils.data import Dataset , DataLoader
from self_attention_later import self_attention

class DataPreprocess():
    def __init__(self, text_file_path):
        self.file_path = text_file_path
        self.vocab = None
        self.vocab_decode = None
        self.word_embedding = None
        self.read_file()
        
        

    
    def read_file(self):
        with open(self.file_path, 'r') as file:
            data = file.read()
            data = re.split(r'[.,!?;:\s]+', data)
            self.vocab , self.vocab_decode = self.create_vocab(data)
            self.word_embedding = self.create_word_embedding(data)
            

    def create_vocab(self, data):
        vocab = set(data)
        vocab_encode = {}
        vocab_decode = {}
        for i, word in enumerate(vocab):
            vocab_encode[word] = i
            vocab_decode[i] = word
        return vocab_encode , vocab_decode
    
    def create_word_embedding(self , data ):

        embeddings = []
        for word in data:
            embeddings.append(self.vocab[word])
        return embeddings
    
    def decode_word_embedding(self):
        text = []
        
        for word_id in self.word_embedding:
            text.append(self.vocab_decode[word_id])
        
        ret_text = " ".join(text)
        ret_text = re.sub(r'\s+([,.?!"()\'])' , r'\1' , ret_text)
        return ret_text

        

encoding = get_encoding('r50k_base')


class Dataload(Dataset):

    def __init__(self  , input_text_path , max_length , stride , train = True , train_test_split = 0.9):
        self.encoding = get_encoding('gpt2')
        self.train = train
        self.ttsplit = train_test_split
        self.labels = []
        self.targets = []
        self.encode_text(input_text_path, max_length , stride)
        self.embeddings = torch.nn.Embedding(50000, 256)
        self.pos_embeddings = torch.nn.Embedding(max_length ,256)(torch.arange(max_length))
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        
        return self.targets[idx] ,  self.labels[idx]

    def encode_text(self, text_file_path , max_length , stride):
        with open(text_file_path, 'r') as f:
            
            text = f.read()
            if( self.train):
                text = text[:1000]#int(self.ttsplit*len(text))]
            else:
                text = text[len(text)-500:]#int(self.ttsplit*len(text)):]
            text_encodes =  self.encoding.encode(text)
            print( f"this is the text encoding length {len(text_encodes)} here is the split {int(len(text)*self.ttsplit)} and text length {len(text)}")
            for i in range(0 , len(text_encodes) - max_length , stride):
                targets = text_encodes[i:i+max_length]
                labels = text_encodes[i+1: i + max_length+1]
                self.labels.append(torch.tensor(labels))
                self.targets.append(torch.tensor(targets))



