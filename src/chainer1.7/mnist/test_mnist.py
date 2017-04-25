#!/usr/bin/env python
#coding: utf-8

"""Chainer example: train a multi-layer perceptron on MNIST
This is a minimal example to write a feed-forward net.

python test_mnist.py ../../../data/img/mnist0.png -g 0


python test_mnist.py ../../../data/img/mnist0.png -g 0 -m mnist.model -r mnist.state
python test_mnist.py ../../../data/img/mnist0.png -g 0 -m mnist_mlp.model

"""
from __future__ import print_function
import argparse

import numpy as np
import six
import os

import chainer
from chainer import computational_graph
from chainer import cuda
import chainer.links as L
import chainer.functions as F
from chainer import optimizers
from chainer import serializers

import data
import time
import net

parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('dataset', help='Path to validation image-label list file')

parser.add_argument('--initmodel', '-m', default='',
                    help='Initialize the model from given file')
parser.add_argument('--resume', '-r', default='',
                    help='Resume the optimization from snapshot')
parser.add_argument('--net', '-n', choices=('simple', 'parallel'),
                    default='simple', help='Network type')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--epoch', '-e', default=1, type=int,
                    help='number of epochs to learn')
parser.add_argument('--unit', '-u', default=1000, type=int,
                    help='number of units')
parser.add_argument('--batchsize', '-b', type=int, default=1,
                    help='learning minibatch size')
args = parser.parse_args()

batchsize = args.batchsize
n_epoch = args.epoch
n_units = args.unit

print('GPU: {}'.format(args.gpu))
print('# unit: {}'.format(args.unit))
print('# Minibatch-size: {}'.format(args.batchsize))
print('# epoch: {}'.format(args.epoch))
print('Network type: {}'.format(args.net))
print('')


# Prepare dataset
print('load MNIST dataset')
import Image
img = np.asarray(Image.open(args.dataset), dtype=np.float32)
#print("img:", img.shape)
items = []
for v in img:
    for item in v:
        items.append(item)
items2 = []
items2.append(items)
x_test = np.array(tuple(items2))
#print("x_test:", x_test.shape)

y_test = np.asarray([0], dtype=np.int32)

start = time.time()


# Prepare multi-layer perceptron model, defined in net.py
if args.net == 'simple':
    #model = L.Classifier(net.MnistMLP(784, n_units, 10))
    model = net.MnistMLP(784, n_units, 10)
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()
    xp = np if args.gpu < 0 else cuda.cupy
elif args.net == 'parallel':
    cuda.check_cuda_available()
    model = L.Classifier(net.MnistMLPParallel(784, n_units, 10))
    xp = cuda.cupy

# Setup optimizer
optimizer = optimizers.Adam()
optimizer.setup(model)

# Init/Resume
if args.initmodel:
    print('Load model from', args.initmodel)
    serializers.load_npz(args.initmodel, model)
if args.resume:
    print('Load optimizer state from', args.resume)
    serializers.load_npz(args.resume, optimizer)



# evaluation                                                                                                                                              
sum_accuracy = 0
sum_loss = 0
x = chainer.Variable(xp.asarray(x_test),
                     volatile='on')
t = chainer.Variable(xp.asarray(y_test),
                     volatile='on')
#print("x shape:",x.shape)
#print("t shape:",t.shape)

y = model(x)
score = F.softmax(y)

print(np.argmax(y.data[0].tolist()))


elapsed_time = time.time() - start
print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
