import scipy as sp
import numpy as np
import random
from scipy.io import loadmat, savemat

FULL_TRAIN_SIZE = 463715
SMALL_TRAIN_SIZE = 5000
SMALL_TEST_SIZE = 2000

infile = loadmat("../MSdata.mat")
trainx = np.asarray(infile["trainx"])
trainy = np.asarray(infile["trainy"])
testx = np.asarray(infile["testx"])

train_idx = np.random.permutation(FULL_TRAIN_SIZE)
small_trainx, small_trainy = trainx[train_idx[:SMALL_TRAIN_SIZE], :], trainy[train_idx[:SMALL_TRAIN_SIZE], :]
small_testx, small_testy = trainx[train_idx[(FULL_TRAIN_SIZE - SMALL_TEST_SIZE):], :], trainy[train_idx[(FULL_TRAIN_SIZE-SMALL_TEST_SIZE):], :]
print(small_trainx.shape)
print(small_testx.shape)
savemat('./data/sm_train.mat', {'trainx':small_trainx, 'trainy':small_trainy})
savemat('./data/sm_test.mat', {'testx':small_testx, 'testy':small_testy})
