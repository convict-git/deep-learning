#! /usr/bin/python3
# author : Priyanshu Shrivastav (from IIT Palakkad, India)

import numpy as np, matplotlib.pyplot as plt
import csv, sys
from sklearn.preprocessing import MinMaxScaler

DATA_SIZE       = 10000
FEATURE_COUNT   = 4
TRAIN_DATA_SET  = "Dataset Question2.csv"
ALPHA           = 0.1

train_sz, test_sz, loss = 0, 0, []
X = np.ndarray((DATA_SIZE, FEATURE_COUNT + 1)) # +1 for bias
Y = np.zeros(DATA_SIZE, dtype=int)
W = np.ndarray(FEATURE_COUNT + 1)
min_max_scaler = MinMaxScaler()

class Perceptron:
# Take training input data
    def take_training_input(self):
        global train_sz, DATA_SIZE, test_sz
        train_sz = 0
        with open(TRAIN_DATA_SET) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                for p in range(FEATURE_COUNT):
                    X[train_sz][p] = float(row[p])
                X[train_sz][FEATURE_COUNT] = 1.0
                Y[train_sz] = int(row[FEATURE_COUNT])
                if Y[train_sz] == 0:
                    Y[train_sz] = -1
                train_sz += 1
        DATA_SIZE = train_sz
        train_sz = int(DATA_SIZE * 0.8) # for training data set 8:2
        test_sz = DATA_SIZE - train_sz

    def scale_input(self):
        global X, min_max_scaler
        X = min_max_scaler.fit_transform(X)

# plot training data data
    def show_data(self, title, xlab="", ylab=""):
        print('=== Data points given to train : {sz} ==='.format(sz=train_sz))
        plt.title(title)
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        for train_data_index in range(train_sz):
            if (Y[train_data_index] == 1):
                plt.plot(X[train_data_index][0], X[train_data_index][1], 'bo')
            else:
                plt.plot(X[train_data_index][0], X[train_data_index][1], 'ro')

# find weights
    def find_weights(self):
        global train_sz, W, loss
        print("================================       TRAINING       ==================================")
        print("Training data size : {tz}".format(tz=train_sz))
        W = np.random.random(FEATURE_COUNT + 1)
        ok = False
        epoch, err = 0, 0
        while ok is False:
            err = 0
            if epoch > 500: # limits on epoch
                break
            W_old, x_add = np.copy(W), []
            epoch += 1
            ok = True
            for tc in range(train_sz):
                y_hat, y_real = float(np.dot(W, X[tc].T)), Y[tc]
                if (y_hat > 0 and y_real < 0):
                    W = np.add(W, -1*ALPHA * X[tc].T)
                    ok = False
                    err += 1
                elif (y_hat < 0 and y_real > 0):
                    W = np.add(W, ALPHA * X[tc].T)
                    ok = False
                    err += 1
            loss.append(err)

            err_pos, err_neg, tp, tn = 0, 0, 0, 0
            for tc in range(train_sz):
                y_hat, y_real = float(np.dot(W, X[tc].T)), Y[tc]
                if (y_hat > 0 and y_real < 0):
                    err_neg += 1
                elif (y_hat < 0 and y_real > 0):
                    err_pos += 1
                elif (y_real > 0):
                    tp += 1
                elif (y_real < 0):
                    tn += 1

            print("----------------------------------------------------------------------------------------")
            print("[ Epoch : {epoch}, Error : {err} ]".format(epoch=epoch, err=err))
            print("W_old : {w_old}".format(w_old=W_old))
            print("W_new : {w_new}".format(w_new=W))
            print("True positive : {tp}, False positive : {fp}".format(tp=tp, fp=err_neg))
            print("True negative : {tn}, False negative : {fn}".format(tn=tn, fn=err_pos))

    def test(self):
        global test_sz, X, Y, loss, W, DATA_SIZE, train_sz

        print("================================       TESTING        ==================================")
        print("Total Data size: ", DATA_SIZE, ", Test Data size : ", test_sz)
        loss, err_pos, err_neg, tp, tn = [], 0, 0, 0, 0
        for tc in range(DATA_SIZE - test_sz, DATA_SIZE):
            y_hat, y_real = float(np.dot(W, X[tc].T)), Y[tc]
            if (y_hat > 0 and y_real < 0):
                err_neg += 1
            elif (y_hat < 0 and y_real > 0):
                err_pos += 1
            elif (y_real > 0):
                tp += 1
            elif (y_real < 0):
                tn += 1
        print("True positive : {tp}, False positive : {fp}".format(tp=tp, fp=err_neg))
        print("True negative : {tn}, False negative : {fn}".format(tn=tn, fn=err_pos))
        print("============================         TESTING FINISHED          =========================")

    def draw_error_count(self, title, sz):
        global loss
        ep = [epoch + 1 for epoch in range(len(loss))]
        accuracy = [sz - x for x in loss]
        plt.title(title + " : Classification Error v/s iterations")
        plt.ylabel("Classification Error")
        plt.xlabel("Iterations")
        plt.plot(ep, loss, 'r')
        plt.show()

        plt.title(title + " : Classification Accuracy v/s iterations")
        plt.ylabel("Classification Accuracy")
        plt.xlabel("Iterations")
        plt.plot(ep, accuracy, 'b')
        plt.show()

t = Perceptron()
t.take_training_input()
t.scale_input()
t.find_weights()
t.draw_error_count("TRAINING", train_sz)
t.test()
