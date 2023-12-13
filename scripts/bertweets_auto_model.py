import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import evaluate


if torch.cuda.is_available():
    device = "cuda:0"
    print("Using GPU")
else:
    device = "cpu"
    print("Using CPU")
    
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False, normalization=True)



class WordDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab_size = 0
        self.max_seq_len = 0
        self._build_vocab()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        words = self.data.iloc[idx]['text']
        party = self.data.iloc[idx]['label']
        
        return torch.tensor(tokenizer.encode(words, padding="max_length", truncation=True), device=device), torch.tensor(party, device=device)
    
    def _build_vocab(self):
        for i, row in self.data.iterrows():
            words = row['text'].split()
            self.max_seq_len = max(self.max_seq_len, len(words))
            for word in words:
                if word not in self.word_to_idx:
                    self.word_to_idx[word] = self.vocab_size
                    self.idx_to_word[self.vocab_size] = word
                    self.vocab_size += 1
                    
    def sample(self, frac, random_state):
        return self.data.sample(frac=frac, random_state=random_state)

from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler

model = AutoModelForSequenceClassification.from_pretrained("vinai/bertweet-base", num_labels=2).to(device)


    
data = pd.read_csv('../data/train_set.csv', encoding='utf-8')
# data = data.sample(frac=0.2, random_state=42)
dataset = WordDataset(data)

# Split data into train and test sets
train_data = dataset.sample(frac=0.7, random_state=42)

dev_data = data.drop(train_data.index)
test_data = dev_data.sample(frac=0.5, random_state=42)
dev_data = dev_data.drop(test_data.index)



hidden_size = 100
num_epochs = 10
batch_size = 32

train_dataset = WordDataset(train_data)
dev_dataset = WordDataset(dev_data)
test_dataset = WordDataset(test_data)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")


optimizer = AdamW(model.parameters(), lr=5e-5)

metric = evaluate.load("accuracy")



num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

from tqdm.auto import tqdm


model.train()
for epoch in range(num_epochs):
    for i, data in enumerate(tqdm(train_dataloader)):
        inputs, labels = data
        attention_mask = torch.tensor([[float(i>0) for i in ii] for ii in inputs], device=device)
        outputs = model(inputs, labels=labels, attention_mask=attention_mask)
        loss = outputs.loss
        loss.backward()
        
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=labels)
        

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        
    metric.compute()
        
    
        
        

model.eval()
for batch in test_dataloader:
    inputs, labels = batch
    with torch.no_grad():
        outputs = model(inputs, labels=labels)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()