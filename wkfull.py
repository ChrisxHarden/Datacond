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

max_seq_length=128

data_collator = DataCollatorWithPadding(tokenizer=tokenizer,padding='max_length',max_length=max_seq_length)

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
    
        



tokens_train=tokenize(trainset)
tokens_val=tokenize(valset)
tokens_test=tokenize(testset)

  
      
    

train_dl = DataLoader(dataset=tokens_train,batch_size=32,collate_fn=data_collator,shuffle=True)
val_dl = DataLoader(dataset=tokens_val,batch_size=32,collate_fn=data_collator)
test_dl=DataLoader(dataset=tokens_test,batch_size=32,collate_fn=data_collator)

# i=1
# for batch in train_dl:
    
#     i-=1
#     if i<0:
#         break
#     else:
#         print(batch["input_ids"].shape)
        


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



def evaluate():
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(batch["input_ids"], labels=batch["input_ids"])

        losses.append(accelerator.gather(outputs.loss))
    loss = torch.mean(torch.cat(losses))
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")
    return loss.item(), perplexity.item()



accelerator = Accelerator()

model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dl, val_dl
)



num_train_epochs = 1
num_update_steps_per_epoch = len(train_dl)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=1_000,
    num_training_steps=num_training_steps,
)


gradient_accumulation_steps = 8
eval_steps = 5000

model.train()

completed_steps = 0
for epoch in range(num_train_epochs):
    for step, batch in tqdm(
        enumerate(train_dataloader, start=1), total=num_training_steps
    ):
        output = model(batch["input_ids"],attention_mask=batch["attention_mask"]).logits
        target=batch["input_ids"][:,1:].contiguous()
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
            eval_loss, perplexity = evaluate()
            accelerator.print({"loss/eval": eval_loss, "perplexity": perplexity})
            model.train()
            