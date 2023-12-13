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
        
        return words, torch.tensor(party, device=device)

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

class Model(nn.Module):
    def __init__(self, token_dim, hidden_size, sequence_length=128):
        super(Model, self).__init__()
        self.bertweet = AutoModel.from_pretrained("vinai/bertweet-base")
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=True, normalization=True)
        self.linear1 = nn.Linear(token_dim, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)
        

    def forward(self, x):
        
        # with torch.no_grad():
        input_ids = torch.tensor([self.tokenizer.encode(x_, padding="max_length") for x_ in x], device=device)
        attention_mask = torch.tensor([[float(i>0) for i in ii] for ii in input_ids], device=device)
        features = self.bertweet(input_ids, attention_mask=attention_mask).last_hidden_state
        
        
        output = self.linear1(features.mean(dim=1))
        output = self.relu(output)
        output = self.linear2(output)
        output = output.squeeze()
        return output
    
    
data = pd.read_csv('../data/train_set.csv', encoding='utf-8')
data = data.sample(frac=0.2, random_state=42)
dataset = WordDataset(data)

hidden_size = 100
num_epochs = 10
batch_size = 32

# Split data into train and test sets
train_data = dataset.sample(frac=0.7, random_state=42)

dev_data = data.drop(train_data.index)
test_data = dev_data.sample(frac=0.5, random_state=42)
dev_data = dev_data.drop(test_data.index)

# Create dataset and dataloader
train_dataset = WordDataset(train_data)
dev_dataset = WordDataset(dev_data)
test_dataset = WordDataset(test_data)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = Model(768, hidden_size).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)

loss_fn = nn.BCEWithLogitsLoss()

metric = evaluate.load("accuracy")

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(tqdm(train_dataloader)):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels.float())
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()

    last_loss = running_loss / len(train_dataloader)
    return last_loss

# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/transformer_train_{}'.format(timestamp))
epoch_number = 0

best_vloss = 1_000_000.

for epoch in range(num_epochs):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)


    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    
    # Disable gradient computation and reduce memory consumption.
    
    with torch.no_grad():
        for i, vdata in enumerate(tqdm(test_dataloader)):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            predicted = torch.round(torch.sigmoid(voutputs))
            vloss = loss_fn(voutputs, vlabels.float())
            running_vloss += vloss
            metric.add_batch(predictions=predicted, references=vlabels)
        
        accuracy = metric.compute()
            

    avg_vloss = running_vloss / len(test_dataloader)
        


    
    print('LOSS train {} valid {} test accuracy {}'.format(avg_loss, avg_vloss, accuracy))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1