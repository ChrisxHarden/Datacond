import portalocker
import os
import time
import copy
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtext.datasets import WikiText2
from torch.utils.data import DataLoader,Dataset
from transformers import GPT2Tokenizer,GPT2Config,GPT2LMHeadModel,DataCollatorWithPadding,get_scheduler
#from utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam, match_loss, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug
''' MLP '''
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc_1 = nn.Linear(128, 128)
        self.fc_2 = nn.Linear(128, 128)
        self.fc_3 = nn.Linear(128, 128)

    def forward(self, x):
        out = x.to(torch.float32).view(x.size(0), -1)
        out = F.relu(self.fc_1(out))
        out = F.relu(self.fc_2(out))
        out = self.fc_3(out)
        return out
    
    


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")##use gpt-2 pretrained-tokenizer 
tokenizer.pad_token=tokenizer.eos_token
data_collator = DataCollatorWithPadding(tokenizer=tokenizer,padding='max_length',max_length=128)

max_seq_length=128
data_size=120
Iteration=10
lr_syn=1.0
init_type='real'
data_save=[]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_path="./data"

# train_file=os.path.join(data_path,"full_train.txt")
# test_file=os.path.join(data_path,"full_test.txt")

raw_train,raw_test=WikiText2(split=("train","test"))

def tokenize(data):
    token_return=[]
    for line in data:
        tokens = tokenizer(line,
        
        truncation=True,
        max_length=max_seq_length,
        padding=True,
        return_overflowing_tokens=True,
        return_length=True
        )
        input_ids=tokens["input_ids"]
        attention_mask=tokens["attention_mask"]
        
        
        
        token_return.append({"input_ids":input_ids,"attention_mask":attention_mask}) 
    return token_return

tokens_train=tokenize(raw_train)
tokens_test=tokenize(raw_test)



random_indices = [random.randint(0, len(tokens_train)) for _ in range(data_size)]

tokens_syn=[tokens_train[data_point] for data_point in random_indices]


train_dl = DataLoader(dataset=tokens_train,batch_size=32,collate_fn=data_collator,shuffle=True)
test_dl=DataLoader(dataset=tokens_test,batch_size=32,collate_fn=data_collator)


eof_num=tokenizer.convert_tokens_to_ids(tokenizer.eos_token)


syn_sentences=[[eof_num]*128]*120


for i,batch in enumerate(tokens_syn):
    syn_sentences[i][:len(batch["input_ids"])]=batch["input_ids"][:]
    
syn_sentences=torch.tensor(syn_sentences,dtype=torch.float32,requires_grad=True)
    
''' training '''
optimizer_syn = torch.optim.SGD([syn_sentences, ], lr=lr_syn, momentum=0.5) 
optimizer_syn.zero_grad()

for it in range(Iteration+1):
    
    model=MLP()
    model.train()
    for param in list(model.parameters()):
        param.requires_grad = False
        
    
    loss_avg = 0

    ''' update synthetic data '''
   
    loss = torch.tensor(0.0).to(device)
    
   
    
    for step,batch in enumerate(train_dl):
        sentence_real=batch["input_ids"]
        #output_real = model(sentence_real).detach()
        output_real = model(sentence_real)
        output_syn = model(syn_sentences)

        #using the distance between 2 vectors

        loss += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0))**2)


        #using cosine similarity


        optimizer_syn.zero_grad()
        loss.backward(retain_graph=True)
        optimizer_syn.step()



    
    if it == Iteration: # only record the final results
        data_save.append(copy.deepcopy(syn_sentences.detach().cpu()))
        print("save the data")
        print(data_save[0].shape)
        torch.save(data_save[0],'data_condensed.pt')
