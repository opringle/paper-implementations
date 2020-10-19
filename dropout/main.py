"""
Dropout from scratch in pytorch
https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import logging


class Dropout(nn.Module):
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()
        self.p = p

    def forward(self, x):
        # return a matrix of 0's and 1's to denote which weights to drop out
        if self.training:
            # get a distribution for the probability a neuron should exist
            binomial = torch.distributions.binomial.Binomial(probs=1 - self.p)
            sample = binomial.sample(x.size())
            # the output of the layer is multiplied during training (inverted dropout)
            # this means each thinned network outputs larger values in training
            scaled_sample = sample * (1.0 / (1 - self.p))
            # multiply the input to the layer by the scaled sample
            return x * scaled_sample
        return x


class DropoutModel(nn.Module):
    def __init__(self):
        # initialize the parent class
        super(DropoutModel, self).__init__()
        self.a = nn.ReLU()
        self.s = nn.Softmax(dim=1)
        self.fc1 = nn.Linear(in_features=5, out_features=3, )
        self.dropout = Dropout(p=0.5)
        self.output_layer = nn.Linear(in_features=3, out_features=2)

    def forward(self, x):
        logging.debug("Input feature shape  = {}".format(x.shape))
        # get an array of 0's and 1's with probability of an element = 1 as dropout prob
        x = self.fc1(x)
        x = self.a(x)
        x = self.dropout(x)
        logging.debug("Feature shape after first layer = {}".format(x.shape))
        x = self.output_layer(x)
        x = self.s(x)
        logging.debug("Feature shape after output layer = {}".format(x.shape))
        return x


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    batch_size = 8
    num_features = 5
    X = torch.rand(batch_size, num_features)
    net = DropoutModel()
    Y_hat = net(X)
    assert Y_hat.shape[1] == 2 and Y_hat.shape[0] == batch_size, "Model output wrong shape!"
    print("Unit test passed!")
