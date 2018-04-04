#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 15:53:38 2018

@author: guanmingqiao
"""

import scipy.io as sio
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor
import numpy as np
from sklearn.preprocessing import RobustScaler
#from sklearn.kernel_approximation import RBFSampler
import math

TRAIN_DIR = "data/md_train.mat"
TEST_DIR = "data/md_test.mat"
CURRENT_TRAIN_SIZE = 50000
TEST_SIZE = 10000

def main (gamma, c, epsilon, train_x, train_y, test_x, test_y, scaler_x, scaler_y):
    n_estimators = 20
    clf = BaggingRegressor(SVR(C = c, epsilon = epsilon, kernel = "rbf", gamma = gamma), max_samples=1 / n_estimators, n_estimators=n_estimators)
    clf.fit(train_x,train_y)
    years = clf.predict(scaler_x.transform(test_x))
    years = scaler_y.inverse_transform(years.reshape(-1, 1)).ravel()
    diff = 0
    for (i, j) in zip(years, test_y):
        diff += abs(i-j)
    diff /= TEST_SIZE
    print ("MSE is: " + str(diff))

    return diff, clf

def predict(clf, train_x, train_y, test_x, scaler_x, scaler_y):
    clf.fit(train_x, train_y)
    years = clf.predict(scaler_x.transform(test_x))
    years = scaler_y.inverse_transform(years.reshape(-1, 1)).ravel()

    print ("Writing.......")
    file = open('result_SVR.csv','w')
    file.write("dataid,prediction\n")
    for i in range(len(years)):
        file.write("{}, {} \n".format(i + 1, years[i]))
    file.close()

if __name__ == "__main__":
    C = [1]
    epsilons = [ 0.01]
    gammas = [0.01]
    best_C, best_epsilon, best_gamma, best_clf = None, None, None, None
    diff = math.inf

    train_x, train_y, small_test_x, big_test_x, test_y = (sio.loadmat(TRAIN_DIR)['trainx'], sio.loadmat(TRAIN_DIR)['trainy'], sio.loadmat(TEST_DIR)['testx'], sio.loadmat('../../MSdata.mat')['testx'], sio.loadmat(TEST_DIR)['testy'])
    test_y = test_y.ravel()

    scaler_x = RobustScaler()
    train_x = scaler_x.fit_transform(train_x)

    scaler_y = RobustScaler()
    train_y = scaler_y.fit_transform(train_y.reshape(-1, 1)).ravel()


    for gamma in gammas:
        for c in C:
            for epsilon in epsilons:
                print("testing SVR model with c = " + str(c) + " and epsilon = " + str(epsilon) + " and gamma = " + str(gamma))
                temp, clf = main(gamma, c, epsilon, train_x, train_y, small_test_x, test_y, scaler_x, scaler_y)
                if temp < diff:
                    diff = temp
                    best_c = c
                    best_epsilon = epsilon
                    best_gamma = gamma
                    best_clf = clf
    print("optimal C is: "+str(best_c)+" and epsilon is: "+str(best_epsilon) + " and gamma is:" + str(best_gamma))
    predict(best_clf, train_x, train_y, big_test_x, scaler_x, scaler_y)
