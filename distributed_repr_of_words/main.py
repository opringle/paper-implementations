"""

https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf
"""
import torch
import torch.nn as nn
import logging
import random

class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super(SkipGramModel, self).__init__()
        self.embedding_size = embedding_size
        self.fc = nn.Linear(in_features=vocab_size, out_features=embedding_size)
        self.output_layer = nn.Linear(in_features=embedding_size, out_features=vocab_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc(x)
        x = self.output_layer(x)
        x = self.softmax(x)
        return x


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    # logging.debug("Unit test successful!")

    sentences = [
        "the quick brown fox jumped over the red fence",
        "i really want to work for quora",
        "hard problems are more fun"
    ]
    # build mapping from a word to its index in the vocabulary
    tokenized_sentences = [
        s.split(" ") for s in sentences
    ]
    tokens = [
        token for sentence in tokenized_sentences for token in sentence
    ]
    token_to_idx = {
        token: idx for idx, token in enumerate(list(set(tokens)))
    }
    # idx_to_token = {
    #     idx: token for token, idx in enumerate(list(set(tokens)))
    # }
    vocab_size = len(token_to_idx)
    logging.debug("Vocab size = {}".format(vocab_size))

    # generate skip grams
    WINDOW_SIZE = 5

    context_target_pairs = []
    for tokenized_sentence in tokenized_sentences:

        # randomly choose a context word in the sentence
        context_idx = random.randint(0, len(tokenized_sentence)-1)
        context_word = tokenized_sentence[context_idx]

        # randomly choose a word that falls within a window of the context
        low = max(0, context_idx - WINDOW_SIZE // 2)
        high = min(len(tokenized_sentence), context_idx + WINDOW_SIZE // 2)
        window_arr = tokenized_sentence[low:context_idx] + tokenized_sentence[context_idx+1:high+1]
        target_idx = random.randint(0, len(window_arr)-1)
        target_word = window_arr[target_idx]
        context_target_pairs.append((context_word, target_word))
    print(context_target_pairs)

    # build the data for the network
    X = torch.empty(size=(len(context_target_pairs), vocab_size))
    Y = torch.empty(size=(len(context_target_pairs), vocab_size))
    for idx, (context, target) in enumerate(context_target_pairs):
        context_idx = token_to_idx[context]
        target_idx = token_to_idx[target]
        x = torch.zeros(size=(vocab_size, ))
        x[context_idx] = 1
        y = torch.zeros(size=(vocab_size,))
        y[target_idx] = 1
        X[idx] = x
        Y[idx] = y

    # test model output
    model = SkipGramModel(vocab_size=vocab_size, embedding_size=10)
    Y_hat = model(X)
    print("Model output = {}".format(Y_hat))
    print(torch.sum(Y_hat, dim=1))

    # fit model to data
