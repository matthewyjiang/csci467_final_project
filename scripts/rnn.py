import pandas as pd
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchtext.vocab import vocab
from tqdm import tqdm
from glove_embedding import get_embeddings

# Functions for Data Loading and Preprocessing
def load_data(file_path, fraction = 1.0):
    df = pd.read_csv(file_path)
    # Sample a fraction of the data randomly
    df_sampled = df.sample(frac=fraction)
    tweets = df_sampled['text'].tolist()
    labels = df_sampled['label'].tolist()
    return tweets, labels

# Function for Data Preprocessing
def preprocess_data(tweets, labels, tokenizer, vocab_obj, max_length):
    processed_data = []
    pad_index = vocab_obj['<pad>']

    for tweet, label in zip(tweets, labels):
        tokenized_tweet = [vocab_obj[token] for token in tokenizer(tweet)]
        if len(tokenized_tweet) < max_length:
            tokenized_tweet += [pad_index] * (max_length - len(tokenized_tweet))
        else:
            tokenized_tweet = tokenized_tweet[:max_length]
        processed_data.append((tokenized_tweet, label_pipeline(label)))
    return processed_data
# Function for Converting Text to Indices
def text_pipeline(x, tokenizer, vocab):
    stoi = vocab.get_stoi()  # Get the string-to-index mapping
    return [stoi.get(token, stoi.get('<unk>')) for token in tokenizer(x)]  # Use '<unk>' index for unknown tokens

def label_pipeline(x):
    return int(x)

# RNN Model Class
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, linear_dim, output_dim):
        super(RNNClassifier, self).__init__()
        # vocab_npa, embs_npa, pad_emb_npa, unk_emb_npa = get_embeddings('../data/glove.twitter.27B.100d.txt')
        # self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(embs_npa).float(), padding_idx=0)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # self.rnn = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, num_layers=2)
        self.rnn = nn.LSTM(embed_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, linear_dim)
        self.output_layer = nn.Linear(linear_dim, output_dim)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, text):
        # with torch.no_grad():
        embedded = self.embedding(text)
        output, (hidden, cell) = self.rnn(embedded)
        # hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        hidden = hidden.squeeze(0)
        out = self.fc1(hidden)
        out = self.sigmoid(out)
        out = self.dropout(out)
        out = self.output_layer(out)
        return out

# Dataset Class
class TweetDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tweet, label = self.data[idx]
        return torch.tensor(tweet), torch.tensor(label)

# Training Function
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, vocab_obj):
    for epoch in range(num_epochs):
        model.train()
        # Initialize the progress bar
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs}')
        for i, (texts, labels) in progress_bar:
            texts, labels = texts.t(), labels.type(torch.FloatTensor)
            optimizer.zero_grad()
            output = model(texts)
            loss = criterion(output.squeeze(1), labels)
            loss.backward()
            optimizer.step()

            # Update the progress bar with the loss information
            progress_bar.set_postfix(loss=loss.item())
            
        # evaluate the model on the train set
        # train_acc = evaluate_model(model, test_loader, vocab_obj)
        # print(f'Train Accuracy: {train_acc:.4f}')
        
def id_to_string(token_ids, vocab):
    # Get the list of tokens from the vocabulary
    tokens_list = vocab.get_itos()
    pad_index = vocab['<pad>']
    # Convert token ids to tokens while filtering out padding
    return ' '.join(tokens_list[token_id] for token_id in token_ids if token_id != pad_index)

def evaluate_model(model, test_loader, vocab):
    model.eval()
    total_acc, total_count = 0, 0
    false_positives = []
    false_negatives = []
    
    tp_count = 0
    tn_count = 0
    fp_count = 0
    fn_count = 0

    with torch.no_grad():
        for texts, labels in test_loader:
            texts, labels = texts.t(), labels
            predicted = model(texts)
            predicted = (predicted.squeeze(1) > 0).int()

            total_acc += (predicted == labels).sum().item()
            total_count += labels.size(0)

            

            # Analyze false positives and negatives
            for i in range(len(labels)):
                true_label = labels[i].item()
                pred_label = predicted[i].item()
                token_ids = texts[:, i].tolist()

                if true_label == 0 and pred_label == 1:
                    # False Positive
                    false_positives.append((id_to_string(token_ids, vocab), true_label, pred_label))
                    fp_count += 1
                elif true_label == 1 and pred_label == 0:
                    # False Negative
                    false_negatives.append((id_to_string(token_ids, vocab), true_label, pred_label))
                    fn_count += 1
                elif true_label == 0 and pred_label == 0:
                    # True Negative
                    tn_count += 1
                elif true_label == 1 and pred_label == 1:
                    # True Positive
                    tp_count += 1
                    
                    

    # Save false positives and negatives
    df_false_positives = pd.DataFrame(false_positives, columns=['Text', 'True Label', 'Predicted Label'])
    df_false_negatives = pd.DataFrame(false_negatives, columns=['Text', 'True Label', 'Predicted Label'])

    df_false_positives.to_csv( '_false_positives.csv', index=False)
    df_false_negatives.to_csv('_false_negatives.csv', index=False)
    
    print(f'TP: {tp_count}')
    print(f'TN: {tn_count}')
    print(f'FP: {fp_count}')
    print(f'FN: {fn_count}')
    
    return total_acc / total_count

# Main Execution Block
def main():
    # Load data
    print("loading data")
    train_tweets, train_labels = load_data('../data/train_set.csv', fraction=1)
    test_tweets, test_labels = load_data('../data/test_set.csv')
    print("finish loading data")
    # Tokenization and building vocabulary
    tokenizer = get_tokenizer('basic_english')
    counter = Counter()
    for tweet in train_tweets:
        counter.update(tokenizer(tweet))

    # Create a vocabulary object
    vocab_obj = vocab(counter)

    # Add special tokens
    vocab_obj.insert_token('<pad>', 0)
    vocab_obj.insert_token('<unk>', 1)
    vocab_obj.set_default_index(vocab_obj['<unk>'])

    # Set maximum length for tokenized tweets
    max_length = 50  # Choose an appropriate value based on your dataset

    # Preprocess data
    train_data = preprocess_data(train_tweets, train_labels, tokenizer, vocab_obj, max_length)
    test_data = preprocess_data(test_tweets, test_labels, tokenizer, vocab_obj, max_length)

    # Build model and dataloaders
    model = RNNClassifier(len(vocab_obj), 100, 24, 24, 1)
    train_loader = DataLoader(TweetDataset(train_data), batch_size=32, shuffle=True)
    test_loader = DataLoader(TweetDataset(test_data), batch_size=32, shuffle=False)

    # Train and evaluate the model
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())
    train_model(model, train_loader, test_loader, criterion, optimizer, 5, vocab_obj)
    accuracy = evaluate_model(model, test_loader, vocab_obj)
    print(f'Test Accuracy: {accuracy:.4f}')

if __name__ == "__main__":
    main()