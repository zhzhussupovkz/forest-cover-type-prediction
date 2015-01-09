#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import time
import csv
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

def get_data():
    data = []
    with open("./train.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data


def get_ds():
    ds = SupervisedDataSet(10, 1)
    data = get_data()
    for k in data:
        ds.addSample((k.get('Elevation'), k.get('Horizontal_Distance_To_Hydrology'), \
            k.get('Horizontal_Distance_To_Roadways'), k.get('Hillshade_9am'), \
            k.get('Hillshade_Noon'), k.get('Hillshade_3pm'), \
            k.get('Horizontal_Distance_To_Fire_Points'), k.get('Wilderness_Area1'), \
            k.get('Wilderness_Area3'), k.get('Wilderness_Area4')), k.get('Cover_Type'))
    return ds

def train_neural_network():
    ds = get_ds()
    net = buildNetwork(10,3,1, bias=True)
    trainer = BackpropTrainer(net, ds)
    return trainer.trainUntilConvergence()

t = train_neural_network()
print t