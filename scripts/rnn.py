import torch
import math
import pandas as pd
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import unicodedata
import string

n_categories = 2

train_dataset = None
dev_dataset = None
test_dataset = None
    
def categoryTensor(category):
    if(category == "Democrat"):
        li = 0
    else:
        li = 1
    tensor = torch.zeros(1, n_categories)
    tensor[0][li] = 1
    return tensor

import random

# Random item from a list
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

# Get a random category and random line from that category
def randomTrainingPair():
    category, handle, line = randomChoice(train_dataset)
    return category, line


def count_words(strings):
    vocab = {}
    for string in strings:
        for word in string:
            if word not in vocab:
                vocab[word] = 1
            else:
                vocab[word] += 1
    return vocab


# model to learn word vectors
class Embeddings(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(Embeddings, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 64)
        self.linear2 = nn.Linear(64, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs
    
    
import time
import math

#from pytorch rnn tutorial
def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()

if __name__ == "__main__":
    # Load dataset, ExtractedTweets.csv

    training_data = pd.read_csv('../data/ExtractedTweets.csv', encoding='utf-8')
    training_data = training_data.to_numpy()

    #remove half the data
    
    training_data = training_data[:math.floor(training_data.shape[0]/10)]

    # tuple: (party, handle, tweet)

    TRAIN_SIZE = math.floor(training_data.shape[0]*0.7)
    DEV_SIZE = math.floor(training_data.shape[0]*0.1)
    TEST_SIZE = training_data.shape[0]-TRAIN_SIZE-DEV_SIZE
    
    CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right


    train_dataset = torch.utils.data.Subset(training_data, range(TRAIN_SIZE))
    dev_dataset = torch.utils.data.Subset(training_data, range(TRAIN_SIZE, TRAIN_SIZE+DEV_SIZE))
    test_dataset = torch.utils.data.Subset(training_data, range(TRAIN_SIZE+DEV_SIZE, TRAIN_SIZE+DEV_SIZE+TEST_SIZE))

    strings = training_data[:,2]
    # strings = [test_sentence]
    for i in range(len(strings)):
        strings[i] = strings[i].split()
        
    # create hashmap to access index of word in embedding
    
    vocab = count_words(strings)
    vocab_index = {word: i for i, word in enumerate(vocab)}
    
    
    vocab_size = len(vocab)
    print(vocab_size)
    
    
    ngrams = []
    for i in range(len(strings)):
        for j in range(len(strings[i])-CONTEXT_SIZE):
            ngrams.append((strings[i][j:j+CONTEXT_SIZE],strings[i][j+CONTEXT_SIZE]))
    
    print(ngrams[:3])
            
    
    model = Embeddings(vocab_size, 48, 2)
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    print("here")
    

    start = time.time()
    losses = []
    
    count = 0
    
    print(len(ngrams))
    
    for epoch in range(10):
        total_loss = 0
        for context, target in ngrams:
            count+=1
            if(count%100==0):
                print(count)
            
            context_idxs = torch.tensor([vocab_index[w] for w in context], dtype=torch.long)
            model.zero_grad()
            
            log_probs = model(context_idxs)

            loss = criterion(log_probs, torch.tensor([vocab_index[target]], dtype=torch.long))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        losses.append(total_loss)
        print("epoch: ", epoch, " loss: ", total_loss)
    print(losses) 