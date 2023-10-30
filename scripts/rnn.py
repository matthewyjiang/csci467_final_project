import pandas as pd
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
    
    
# embeddings_model = torch.load('embeddings_model.pt')

# Define dataset class
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
        words = self.data.iloc[idx]['Tweet'].split()
        seq_len = len(words)
        seq = [self.word_to_idx[word] for word in words]
        seq += [0] * (self.max_seq_len - seq_len)
        return torch.tensor(seq, device=device), torch.tensor(seq_len, device=device)

    def _build_vocab(self):
        for i, row in self.data.iterrows():
            words = row['Tweet'].split()
            self.max_seq_len = max(self.max_seq_len, len(words))
            for word in words:
                if word not in self.word_to_idx:
                    self.word_to_idx[word] = self.vocab_size
                    self.idx_to_word[self.vocab_size] = word
                    self.vocab_size += 1
                    
    def sample(self, frac, random_state):
        return self.data.sample(frac=frac, random_state=random_state)

# Define RNN model
class WordRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size):
        super(WordRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, seq_len):
        x = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(x, seq_len.cpu(), batch_first=True, enforce_sorted=False)
        output, (h_n, c_n) = self.rnn(packed)
        x = self.fc(h_n.squeeze(0))
        x = self.softmax(x)
        return x

# Load data
data = pd.read_csv('../data/ExtractedTweets_new.csv', encoding='utf-8')

dataset = WordDataset(data)

# Define hyperparameters
vocab_size = len(dataset.word_to_idx)
embedding_dim = 100
hidden_size = 128
output_size = 2
num_epochs = 10
batch_size = 32

# Split data into train and test sets
train_data = dataset.sample(frac=0.8, random_state=42)
from tqdm import tqdm

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

model = WordRNN(vocab_size, embedding_dim, hidden_size, output_size).to(device)
optimizer = optim.Adam(model.parameters())

loss_fn = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch_idx, (x, seq_len) in enumerate(tqdm(train_dataloader)):
        optimizer.zero_grad()
        y = train_dataloader.iloc[batch_idx]['Party']
        
        y = torch.tensor(y, device=device)
        output = model(x.to(device), seq_len)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Evaluate model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (x, seq_len) in enumerate(test_dataloader):
            y = test_data.iloc[batch_idx]['Party']
            if y == 'Democrat':
                y = 0
            else:
                y = 1
            y = torch.tensor(y, device=device)
            output = model(x.to(device), seq_len)
            _, predicted = torch.max(output.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    accuracy = 100 * correct / total
    print('Epoch [{}/{}], Loss: {:.4f}, Test Accuracy: {:.2f}%'.format(epoch+1, num_epochs, train_loss/len(train_dataloader), accuracy))
