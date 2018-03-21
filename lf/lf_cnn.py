import torch
from torch import nn

def convRelu(i, batchNormalization=False, leakyRelu=False):
    nc = 3
    ks = [3, 3, 3, 3, 3, 3, 2]
    ps = [1, 1, 1, 1, 1, 1, 1]
    ss = [1, 1, 1, 1, 1, 1, 1]
    nm = [64, 128, 256, 256, 512, 512, 512]

    cnn = nn.Sequential()

    nIn = nc if i == 0 else nm[i - 1]
    nOut = nm[i]
    cnn.add_module('conv{0}'.format(i),
                   nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
    if batchNormalization:
        cnn.add_module('batchnorm{0}'.format(i), nn.InstanceNorm2d(nOut))
        # cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
    if leakyRelu:
        cnn.add_module('relu{0}'.format(i),
                       nn.LeakyReLU(0.2, inplace=True))
    else:
        cnn.add_module('relu{0}'.format(i), nn.ReLU(True))
    return cnn

def makeCnn():

    cnn = nn.Sequential()
    cnn.add_module('convRelu{0}'.format(0), convRelu(0))
    cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))
    cnn.add_module('convRelu{0}'.format(1), convRelu(1))
    cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))
    cnn.add_module('convRelu{0}'.format(2), convRelu(2, True))
    cnn.add_module('convRelu{0}'.format(3), convRelu(3))
    cnn.add_module('pooling{0}'.format(2), nn.MaxPool2d(2, 2))
    cnn.add_module('convRelu{0}'.format(4), convRelu(4, True))
    cnn.add_module('convRelu{0}'.format(5), convRelu(5))
    cnn.add_module('pooling{0}'.format(3), nn.MaxPool2d(2, 2))
    cnn.add_module('convRelu{0}'.format(6), convRelu(6, True))
    cnn.add_module('pooling{0}'.format(4), nn.MaxPool2d(2, 2))

    return cnn
