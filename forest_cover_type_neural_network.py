#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import time
import csv
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import ClassificationDataSet
from pybrain.supervised.trainers import BackpropTrainer

from pybrain.structure import SoftmaxLayer
from pybrain.utilities import percentError

import pylab as pl
import matplotlib.pyplot as plt

def get_data():
    data = []
    with open("./train.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data


def get_ds():
    ds = ClassificationDataSet(10, 1, nb_classes=7)
    data = get_data()
    for k in data:
        ds.addSample((k.get('Elevation'), k.get('Horizontal_Distance_To_Hydrology'), \
            k.get('Horizontal_Distance_To_Roadways'), k.get('Hillshade_9am'), \
            k.get('Hillshade_Noon'), k.get('Hillshade_3pm'), \
            k.get('Horizontal_Distance_To_Fire_Points'), k.get('Wilderness_Area1'), \
            k.get('Wilderness_Area3'), k.get('Wilderness_Area4')), k.get('Cover_Type'))
    return ds

# plot error
def plot_error(trnError, valError):
    print "Plot train error and value error..."
    f = plt.figure(figsize=(8, 6))
    plt.plot(trnError, 'b', valError, 'r')
    f.savefig('./test_plot/nn_error.png')

# train neural network
def train_neural_network():
    start = time.clock()
    ds = get_ds()

    # split main data to train and test parts
    train, test = ds.splitWithProportion(0.75)

    # build nn with 10 inputs, 3 hidden layers, 1 output neuron
    net = buildNetwork(10,3,1, bias=True)

    # use backpropagation algorithm
    trainer = BackpropTrainer(net, train, momentum = 0.1, weightdecay = 0.01)

    # plot error
    trnError, valError = trainer.trainUntilConvergence(dataset = train, maxEpochs = 50)

    plot_error(trnError, valError)

    print "train the model..."
    trainer.trainOnDataset(train, 500)
    print "Total epochs: %s" % trainer.totalepochs

    print "activate..."
    out = net.activateOnDataset(test).argmax(axis = 1)
    percent = 100 - percentError(out, test['target'])
    print "%s" % percent

    end = time.clock()
    print "Time: %s" % str(end-start)

train_neural_network()