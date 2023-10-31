import argparse
from collections import Counter, defaultdict
import sys
import csv
from tqdm import tqdm
import math

import numpy as np

OPTS = None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation-set', '-e', choices=['dev', 'test', 'newbooks'])
    parser.add_argument('--analyze-counts', '-a', action='store_true')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

def read_data(filename):
    dataset = []
    with open(filename) as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header
        for row in reader:
            if len(row) != 3:  # Check if there are enough values in the row
                print(f"Warning: Skipping line due to unexpected format: {row}")
                continue
            party, handle, tweet = row
            dataset.append((tweet.split(' '), handle))
    return dataset

def get_vocabulary(dataset):
    return list(set(word for (tweet, handle) in dataset for word in tweet))

def get_label_counts(train_data):
    """Count the number of examples with each label in the dataset.

    We will use a Counter object from the python collections library.
    A Counter is essentially a dictionary with a "default value" of 0
    for any key that hasn't been inserted into the dictionary.

    Args:
        train_data: A list of (words, label) pairs, where words is a list of str
    Returns:
        A Counter object mapping each label to a count.
    """
    label_counts = Counter()
    ### BEGIN_SOLUTION 4a
    for word, lable in train_data:
        label_counts[lable]+=1
    ### END_SOLUTION 4a
    return label_counts

def get_word_counts(train_data):
    """Count occurrences of every word with every label in the dataset.

    We will create a separate Counter object for each label.
    To do this easily, we create a defaultdict(Counter),
    which is a dictionary that will create a new Counter object whenever
    we query it with a key that isn't in the dictionary.

    Args:
        train_data: A list of (words, label) pairs, where words is a list of str
    Returns:
        A Counter object where keys are tuples of (label, word), mapped to
        the number of times that word appears in an example with that label
    """
    word_counts = defaultdict(Counter)
    ### BEGIN_SOLUTION 4a
    for words, label in train_data:
        for word in words:
            word_counts[label][word]+=1

    '''for words, label in train_data:
        for word in words:
            word_counts[(word, label)]=0
    for words, label in train_data:
        for word in words:
            word_counts[(word, label)]+=1'''
    ### END_SOLUTION 4a
    return word_counts

def predict(words, label_counts, word_counts, vocabulary):
    """Return the most likely label given the label_counts and word_counts.

    Args:
        words: List of words for the current input.
        label_counts: Counts for each label in the training data
        word_counts: Counts for each (label, word) pair in the training data
        vocabulary: List of all words in the vocabulary
    Returns:
        The most likely label for the given input words.
    """
    labels = list(label_counts.keys())  # A list of all the labels
    ### BEGIN_SOLUTION 4a
    prob = dict()
    for label in labels:
        pi = label_counts[label] / sum(label_counts.values())
        sum1 = sum(word_counts[label].values())
        sum2 = 0
        for word in words:
            sum2 += np.log((word_counts[label][word]+1))-np.log(sum1+len(vocabulary))

        prob[label] = sum2+pi

    return max(prob, key=prob.get)
    ### END_SOLUTION 4a


def evaluate(label_counts, word_counts, vocabulary, dataset, name, print_confusion_matrix=False):
    num_correct = 0
    confusion_counts = Counter()

    # Dictionary to map party integer values to string names
    party_names = {0: 'Democrat', 1: 'Republican'}  # You can update this dictionary based on your data

    for words, label in tqdm(dataset, desc=f'Evaluating on {name}'):
        pred_label = predict(words, label_counts, word_counts, vocabulary)
        confusion_counts[(label, pred_label)] += 1
        if pred_label == label:
            num_correct += 1

    accuracy = 100 * num_correct / len(dataset)

    # Use party names for output instead of integers
    print(f'{name} accuracy: {num_correct}/{len(dataset)} = {accuracy:.3f}%')
    if print_confusion_matrix:
        print(''.join(['actual\\predicted'] + [party_names[label].rjust(12) for label in label_counts]))
        for true_label in label_counts:
            print(''.join([party_names[true_label].rjust(16)] + [
                str(confusion_counts[true_label, pred_label]).rjust(12)
                for pred_label in label_counts]))


def analyze_counts(label_counts, word_counts, vocabulary):
    """Analyze the word counts to identify the most predictive features.

    For each label, print out the top ten words that are most indictaive of the label.
    There are multiple valid ways to define what "most indicative" means.
    Our definition is that if you treat the single word as the input x,
    find the words with largest p(y=label | x).

    The print steps are provided. You only have to compute the defaultdict "p_label_given_word".
    The key of the defaultdict is the label, and the value of the defaultdict is a list of
    probabilities p(y=label | x) of each single word x.
    """
    labels = list(label_counts.keys())  # A list of all the labels
    p_label_given_word = defaultdict(list)
    ### BEGIN_SOLUTION 4b
    total_labels = sum(label_counts.values())
    total_words = len(vocabulary)
    for label in labels:
        for word in vocabulary:
            x = label_counts[label]/total_labels * (word_counts[label][word]+1)/(sum(word_counts[label].values())+total_words)
            y=0
            for label2 in labels:
                y += label_counts[label2]/total_labels * (word_counts[label2][word]+1) / (sum(word_counts[label2].values())+total_words)
            p_label_given_word[label].append((word, x/y))
    ### END_SOLUTION 4b

    for label in labels:
        print(f'Label {label}')
        sorted_scores = sorted(p_label_given_word[label],
                               key=lambda x: x[1], reverse=True)
        for word, p in sorted_scores[:10]:
            print(f'    {word}: {p:.3f}')


def main():
    train_data = read_data('data/train_NB.csv')
    dev_data = read_data('data/dev_NB.csv')
    test_data = read_data('data/test_NB.csv')
    newbooks_data = read_data('data/ExtractedTweets_new.csv')

    vocabulary = get_vocabulary(train_data)  # The set of words present in the training data
    label_counts = get_label_counts(train_data)
    word_counts = get_word_counts(train_data)
    if OPTS.analyze_counts:
        analyze_counts(label_counts, word_counts, vocabulary)
    evaluate(label_counts, word_counts, vocabulary, train_data, 'train')
    if OPTS.evaluation_set == 'dev':
        evaluate(label_counts, word_counts, vocabulary, dev_data, 'dev', print_confusion_matrix=True)
    elif OPTS.evaluation_set == 'test':
        evaluate(label_counts, word_counts, vocabulary, test_data, 'test', print_confusion_matrix=True)
    elif OPTS.evaluation_set == 'newbooks':
        evaluate(label_counts, word_counts, vocabulary, newbooks_data, 'newbooks', print_confusion_matrix=True)

if __name__ == '__main__':
    OPTS = parse_args()
    main()

