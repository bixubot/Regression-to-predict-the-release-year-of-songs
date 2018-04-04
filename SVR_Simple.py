#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 21:06:35 2018

@author: guanmingqiao
"""

import scipy.io as sio
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor
from sklearn.preprocessing import RobustScaler

def predict(c, epsilon, gamma, train_x, train_y, test_x, scaler_x, scaler_y):
    print("doing SVR model with c = " + str(c) + " and epsilon = " + str(epsilon) + " and gamma = " + str(gamma))
    n_estimators = 20
    clf = BaggingRegressor(SVR(C = c, epsilon = epsilon, kernel = "rbf", gamma = gamma), max_samples=1 / n_estimators, n_estimators=n_estimators)
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
    train_x, train_y, test_x = (sio.loadmat('../../MSdata.mat')['trainx'], sio.loadmat('../../MSdata.mat')['trainy'], sio.loadmat('../../MSdata.mat')['testx'])

    scaler_x = RobustScaler()
    train_x = scaler_x.fit_transform(train_x)

    scaler_y = RobustScaler()
    train_y = scaler_y.fit_transform(train_y.reshape(-1, 1)).ravel()

    predict(1, 0.01, 0.01, train_x, train_y, test_x, scaler_x, scaler_y)