"""
https://nlp.stanford.edu/pubs/glove.pdf
"""
import logging
from collections import Counter, defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class GloveDataset:
    def __init__(self, text, n_words=200000, window_size=5):
        self._window_size = window_size
        self._tokens = text.split(" ")[:n_words]
        # count occurences of each token
        word_counter = Counter()
        word_counter.update(self._tokens)
        # build dicts mappings tokens to indices
        self._word_to_idx = {w: i for i, w in enumerate(word_counter)}
        self._idx_to_word = {i: w for w, i in self._word_to_idx.items()}
        self._id_tokens = [self._word_to_idx[word] for word in self._tokens]
        self.vocab_size = len(self._idx_to_word)
        # create the cooccurence matrix
        self._create_coocurrence_matrix()

    def _create_coocurrence_matrix(self):
        cooc_mat = defaultdict(Counter)
        for idx, word in enumerate(self._id_tokens):
            window_start_idx = max(0, idx - self._window_size)
            window_end_idx = min(len(self._id_tokens), window_start_idx + self._window_size + 1)
            for j in range(window_start_idx, window_end_idx):
                # if the index is not the same as the point in the window
                if idx != j:
                    c_idx = self._id_tokens[j]
                    cooc_mat[word][c_idx] += 1 / abs(j-idx)  # weight addition by distance from target

        self._i_idx = list()
        self._j_idx = list()
        self._xij = list()

        # create indices and x values tensors
        for w, cnt in cooc_mat.items():
            for c, v in cnt.items():
                self._i_idx.append(w)
                self._j_idx.append(c)
                self._xij.append(v)

        self._i_idx = torch.LongTensor(self._i_idx)
        self._j_idx = torch.LongTensor(self._j_idx)
        self._xij = torch.FloatTensor(self._xij)

    def get_batches(self, batch_size=3):
        # get random selection of all coocurrence pairs in the matrix
        random_idxs = torch.LongTensor(np.random.choice(len(self._xij), size=len(self._xij), replace=False))

        for p in range(0, len(random_idxs), batch_size):
            batch_ids = random_idxs[p:p+batch_size]
            yield self._xij[batch_ids], self._i_idx[batch_ids], self._j_idx[batch_ids]


class GloveModel(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(GloveModel, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(num_embeddings=self.num_embeddings, embedding_dim=embedding_dim)
        self.biases = nn.Embedding(num_embeddings=self.num_embeddings, embedding_dim=1)

    def forward(self, x):
        logging.debug("Model input = {}".format(x))
        embedding_i = self.embeddings(x[0])
        logging.debug("Embeddings for words i shape = {}".format(embedding_i.shape))
        embedding_j = self.embeddings(x[1])
        logging.debug("Embeddings for words j shape = {}".format(embedding_j.shape))
        dot_product = (embedding_i * embedding_j).sum(dim=1)
        logging.debug("Dot product shape = {}".format(dot_product.shape))
        bias_i = self.biases(x[0]).flatten()
        bias_j = self.biases(x[1]).flatten()
        logging.debug("Bias shape = {}".format(bias_i.shape))
        # TODO: label is logged
        return dot_product + bias_i + bias_j

def weight_func(x, x_max, alpha):
    wx = (x/x_max)**alpha
    wx = torch.min(wx, torch.ones_like(wx))
    return wx


def wmse_loss(weights, pred, label):
    loss = weights * F.mse_loss(pred, label)
    return torch.mean(loss)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    dataset = GloveDataset(text="I love to move it move it. I love to move it!")
    net = GloveModel(
        num_embeddings=dataset.vocab_size,
        embedding_dim=5
    )
    optimizer = optim.Adagrad(net.parameters(), lr=0.05)
    EPOCHS = 100
    ALPHA = 0.75
    X_MAX = 100

    for e in range(1, EPOCHS+1):
        epoch_loss = 0
        for batch_idx, (xij, xi, xj) in enumerate(dataset.get_batches()):
            optimizer.zero_grad()
            pred = net([xi, xj])
            logging.debug("Model pred = {}".format(pred))
            weights = weight_func(alpha=ALPHA, x=xij, x_max=X_MAX)
            label = torch.log(xij)
            logging.debug("label = {}".format(label))
            loss = wmse_loss(weights, pred, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.mean()
        logging.info("Epoch {} loss = {}".format(e, epoch_loss))
    logging.info('Unit test success!')
