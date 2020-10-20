"""

https://cs.stanford.edu/~quocle/paragraph_vector.pdf
"""

import torch
import torch.nn as nn
import torch.optim as optim
import logging
import random
import numpy as np

class DistributedMemoryModel(nn.Module):
    def __init__(self, num_documents, vocab_size, word_embedding_size, document_embedding_size, context_size):
        super(DistributedMemoryModel, self).__init__()
        self.num_documents = num_documents
        self.vocab_size = vocab_size
        self.word_embedding_size = word_embedding_size
        self.document_embedding_size = document_embedding_size
        self.context_size = context_size
        self.word_embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=word_embedding_size
        )
        self.paragraph_embeddings = nn.Embedding(
            num_embeddings=num_documents,
            embedding_dim=document_embedding_size
        )
        self.output_layer = nn.Linear(
            in_features=context_size*word_embedding_size+document_embedding_size,
            out_features=vocab_size
        )
        self.softmax_activation = nn.Softmax(dim=1)

    def forward(self, x):
        logging.debug("Model input shape = {}".format(x.shape))
        paragraph_idxs = x[:, 0]
        logging.debug("Paragraph idxs shape = {}".format(paragraph_idxs.shape))
        paragraph_embedding = self.paragraph_embeddings(paragraph_idxs)
        logging.debug("Paragraph embedding shape = {}".format(paragraph_embedding.shape))
        word_embeddings = self.word_embeddings(x[:, 1:]).flatten(start_dim=1)
        logging.debug("Word embedding shape = {}".format(word_embeddings.shape))
        concatenated_embeddings = torch.cat((paragraph_embedding, word_embeddings), dim=1)
        logging.debug("Concatenated features shape = {}".format(concatenated_embeddings.shape))
        return self.softmax_activation(self.output_layer(concatenated_embeddings))


def preprocess_text(documents):
    tokenized_documents = [
        doc.split() for doc in documents
    ]
    tokens = []
    for tokenized_document in tokenized_documents:
        for token in tokenized_document:
            tokens.append(token)
    distinct_tokens = list(set(tokens))
    m = len(distinct_tokens)
    idx_to_token = {
        idx: value for idx, value in enumerate(distinct_tokens)
    }
    token_to_idx = {
        value: idx for idx, value in enumerate(distinct_tokens)
    }
    logging.debug(
        "{} documents and {} tokens".format(N, m)
    )
    return tokenized_documents, idx_to_token, token_to_idx, m


def build_training_data(tokenized_documents, samples, N, context_size):
    X = []
    Y = []
    for i in range(samples):
        # randomly select a paragraph idx (integer from 0 to N-1)
        paragraph_idx = random.randint(0, N - 1)
        paragraph_length = len(tokenized_documents[paragraph_idx])

        # randomly select a starting point (integer from 0 to len(document)-1-context_size
        low = 0
        high = max(0, paragraph_length - 1 - context_size - 1)
        start = random.randint(low, high)
        end = start + context_size
        tokens = tokenized_documents[paragraph_idx][start:end]
        target = tokenized_documents[paragraph_idx][end]
        # logging.debug("Context tokens = {}\tTarget token = {}".format(tokens, target))
        token_idxs = [
            token_to_idx[t] for t in tokens
        ]
        # logging.debug("Context indices = {}\tTarget index = {}".format(token_idxs, token_to_idx[target]))
        x = np.array([paragraph_idx] + token_idxs)
        y = token_to_idx[target]

        X.append(x)
        Y.append(y)
    X = torch.from_numpy(np.array(X))
    Y = torch.from_numpy(np.array(Y))
    return X, Y


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    documents = [
        "I once tried to start a dating business",
        "In winter I love to ski",
        "I want to work for Quora"
    ]
    N = len(documents)
    p = 5
    q = 12
    context_size = 3

    # preprocess and build vocab
    tokenized_documents, idx_to_token, token_to_idx, m = preprocess_text(documents)
    X, Y = build_training_data(
        tokenized_documents=tokenized_documents,
        samples=100,
        N=N,
        context_size=context_size
    )
    logging.debug("Input feature shape = {}".format(X.shape))

    # define the model
    model = DistributedMemoryModel(
        num_documents=N,
        vocab_size=m,
        word_embedding_size=q,
        document_embedding_size=p,
        context_size=context_size
    )
    Y_hat = model(X)
    logging.debug("Model predictions shape = {}".format(Y_hat.shape))

    # fit the model to learn word and paragraph embeddings
    cost = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    for e in range(100):
        Y_hat = model(X)
        loss = cost(Y_hat, Y)
        logging.info("Epoch {} loss = {}".format(e, loss))
        loss.backward()
        optimizer.step()

    # TODO:
    # fix word embeddings and softmax weights
    # refit the model to learn paragraph embeddings
    # use hierarchical softmax
    # build training data as in paper
    # training loop should use stochastic gradient descent
    # print model accuracy during training
