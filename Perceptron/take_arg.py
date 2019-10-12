#! /usr/bin/python3
# author : Priyanshu Shrivastav (from IIT Palakkad, India)

import numpy as np, matplotlib.pyplot as plt
import csv, sys, argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataSize", type=int, default=10000)
parser.add_argument("--featureCount", type=int, default=2)
parser.add_argument("--dataSetTrain", default="Train1.csv")
parser.add_argument("--dataSetTest", default="Test1.csv")
parser.add_argument("--learningRate", default=0.1)
args = parser.parse_args()

DATA_SIZE = args.dataSize
PARAMETER_COUNT = args.featureCount
TRAIN_DATA_SET  = args.dataSetTrain
TEST_DATA_SET   = args.dataSetTest
ALPHA           = args.learningRate
