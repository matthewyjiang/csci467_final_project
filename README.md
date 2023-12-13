# Classifying Poltical Sentiment on Tweet Samples

CSCI467 Final Project by Matthew Jiang, Haohan Zhang, Yikai Yang

To use GloVe embeddings, download `glove.twitter.27B.100d.txt` from [https://nlp.stanford.edu/projects/glove/](GloVe: Global Vectors for Word Representation), and place it in the `./data/` folder.

Train LSTMs 
```
cd scripts
python3 rnn.py
```

Train Naive Bayes
```
cd scripts
python3 Baseline_Naive_Bayes.py
```