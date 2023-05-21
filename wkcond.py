import torch
from torch import nn,optim
import portalocker
from torchtext.datasets import WikiText2
from transformers import GPT2Tokenizer,GPT2Config,GPT2LMHeadModel,DataCollatorWithPadding,get_scheduler
from torch.utils.data import DataLoader,Dataset
from tqdm import tqdm
from accelerate import Accelerator


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")##use gpt-2 pretrained-tokenizer 
tokenizer.pad_token=tokenizer.eos_token
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

max_seq_length=128

trainset,valset,testset=WikiText2()

        

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
    
    
    
eof_num=tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
synset=torch.load("data_condensed.pt")
synset = torch.clamp(synset, min=0, max=eof_num)
train_dl=DataLoader(dataset=synset,batch_size=4,shuffle=True)


tokens_val=tokenize(valset)
tokens_test=tokenize(testset)

  
val_dl = DataLoader(dataset=tokens_val,batch_size=4,collate_fn=data_collator)
test_dl=DataLoader(dataset=tokens_test,batch_size=16,collate_fn=data_collator)



configuration=GPT2Config(bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id)
model=GPT2LMHeadModel(configuration)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
lossFunc=nn.CrossEntropyLoss()
weight_decay = 0.1


def get_grouped_params(model, no_decay=["bias", "LayerNorm.weight"]):
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [
        {"params": params_with_wd, "weight_decay": weight_decay},
        {"params": params_without_wd, "weight_decay": 0.0},
    ]
    
    
optimizer=optim.AdamW(get_grouped_params(model),lr=0.001)



accelerator = Accelerator()

model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dl, val_dl
)



num_train_epochs = 20
num_update_steps_per_epoch = len(train_dl)

num_training_steps = num_train_epochs * num_update_steps_per_epoch
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=1_000,
    num_training_steps=num_training_steps,
)


gradient_accumulation_steps = 8
eval_steps =1
lowest_it=0
lowest_perplexity=float('inf')
model.train()

completed_steps = 0
for epoch in range(num_train_epochs):
    for step, batch in tqdm(
        enumerate(train_dataloader, start=1), total=num_training_steps
    ):
        cal_batch=batch.to(torch.long)
        output = model(cal_batch.to(device)).logits
        
        target=cal_batch[:,1:].contiguous()
        target=target.view(-1)
        
        
        prediction=output[:,:-1,:].contiguous()
        prediction=prediction.view(-1,prediction.size(-1))
        
        
        loss = lossFunc(prediction,target)
        
        
        if step % 100 == 0:
            accelerator.print(
                { 
                    "steps": completed_steps,
                    "loss/train": loss.item() * gradient_accumulation_steps,
                }
            )
            
            
        loss = loss / gradient_accumulation_steps
        accelerator.backward(loss)
        if step % gradient_accumulation_steps == 0:
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            completed_steps += 1
        if (step % (eval_steps * gradient_accumulation_steps)) == 0:


            model.eval()
            total_loss=0
            iter=0
            
            for step,batch in enumerate(val_dl):
              
              with torch.no_grad():


                outputs = model(batch["input_ids"].to(device),attention_mask=batch["attention_mask"].to(device)).logits


              eval_target=batch["input_ids"][:,1:].contiguous()
              eval_target=eval_target.view(-1)
              #print(eval_target.shape)

              eval_prediction=output[:,:-1,:].contiguous()
              eval_prediction=eval_prediction.view(-1,eval_prediction.size(-1))
              #print(eval_prediction.shape)
 

              if eval_target.shape[0]==eval_prediction.shape[0]:
                loss = lossFunc(eval_prediction.to(device),eval_target.to(device))
                total_loss+=loss.item() 
                iter+=1
              

            print(total_loss)
            if iter!=0:
              total_loss=total_loss/iter
            
            try:
              perplexity = 2**total_loss
            except OverflowError:
              perplexity = float("inf")

            if perplexity<lowest_perplexity:
              lowest_perplexity=perplexity
              lowest_it=epoch
            accelerator.print({"loss/eval": total_loss, "perplexity": perplexity})
            model.train()
            
            
            
print(lowest_perplexity)           