"""
Residual networks solve the problem of vanishing gradients, where training loss actually increases with depth
The keep idea is to introduce skip connections. If you initialize weight layers to zero, then the starting network is
just the identity function.
https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf
"""
import torch
import torch.nn as nn
import torch.functional as F
import logging


class SkipLayer(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features=input_features, out_features=input_features)
        self.a = nn.ReLU()
        self.fc2 = nn.Linear(in_features=input_features, out_features=input_features)

    def forward(self, x):
        logging.debug("Input to skip layer = \n{}".format(x))
        x1 = self.fc1(x)
        logging.debug("Output from first linear layer before activation = \n{}".format(x1))
        x2 = self.a(x1)
        x3 = self.fc2(x2)
        logging.debug("Output from second linear layer before activation = \n{}".format(x3))
        x4 = x3 + x
        logging.debug("After addition with input = \n{}".format(x4))
        return self.a(x4)


class SkipModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.skip1 = SkipLayer(input_features=5)
        self.output_layer = nn.Linear(in_features=5, out_features=2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.skip1(x)
        x = self.output_layer(x)
        x = self.softmax(x)
        return x


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    batch_size = 4
    num_features = 5
    X = torch.rand(batch_size, num_features)
    model = SkipModel()
    Y_hat = model(X)
    logging.debug("Unit test passed!")
