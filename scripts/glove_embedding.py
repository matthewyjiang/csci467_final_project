import numpy as np
from tqdm import tqdm
import os.path

def get_embeddings(file):
    vocab,embeddings = [],[]
    # check for vocab and emb files, if they exist, load them and return
    
    if os.path.isfile('vocab_npa.npy') and os.path.isfile('embs_npa.npy'):
        print('Loading vocab and embeddings from files...')
        with open('vocab_npa.npy','rb') as f:
            vocab_npa = np.load(f)
        with open('embs_npa.npy','rb') as f:
            embs_npa = np.load(f)
        pad_emb_npa = embs_npa[0,:]
        unk_emb_npa = embs_npa[1,:]
        return vocab_npa,embs_npa,pad_emb_npa,unk_emb_npa
            
   
    
        
    print('Loading Glove Embeddings...')
    with open(file, 'r') as f:
        full_content = f.read().strip().split('\n')
    
    
    progress_bar = tqdm(range(len(full_content)), desc='Processing Glove Embeddings')
    
    for i in range(len(full_content)):
        i_word = full_content[i].split(' ')[0]
        i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]
        vocab.append(i_word)
        embeddings.append(i_embeddings)
        progress_bar.update(1)
        
    vocab_npa = np.array(vocab)
    embs_npa = np.array(embeddings)
    vocab_npa = np.insert(vocab_npa, 0, '<pad>')
    vocab_npa = np.insert(vocab_npa, 1, '<unk>')
    pad_emb_npa = np.zeros((1,embs_npa.shape[1]))   #embedding for '<pad>' token.
    unk_emb_npa = np.mean(embs_npa,axis=0,keepdims=True)    #embedding for '<unk>' token.
    
    embs_npa = np.vstack((pad_emb_npa,unk_emb_npa,embs_npa))
    
    
    with open('vocab_npa.npy','wb') as f:
        np.save(f,vocab_npa)

    with open('embs_npa.npy','wb') as f:
        np.save(f,embs_npa)
    return vocab_npa,embs_npa,pad_emb_npa,unk_emb_npa