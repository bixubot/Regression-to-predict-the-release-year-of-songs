import scipy.io as sio
import numpy as np
from sklearn import linear_model
import math

TRAIN_DIR = "data/md_train.mat"
TEST_DIR = "data/md_test.mat"
TEST_SIZE = 10000

def main (alp):
    train_x, train_y, test_x, test_y = (sio.loadmat(TRAIN_DIR)['trainx'], sio.loadmat(TRAIN_DIR)['trainy'], sio.loadmat(TEST_DIR)['testx'], sio.loadmat(TEST_DIR)['testy'])
    train_y = train_y.ravel()
    test_y = test_y.ravel()
    clf = linear_model.Perceptron(penalty='elasticnet',alpha=alp,n_jobs=4, )
    clf.fit(train_x,train_y)
    years = clf.predict(test_x)
    
    diff = float(0.0)
    for (i, j) in zip(years, test_y):
        diff += abs(i-j)
    diff /= TEST_SIZE   
    print ("MSE is: " + str(diff) +" with alpha: "+str(alp))
    return diff
    
def predict(clf):
    test_x = sio.loadmat('../MSdata.mat')['testx']

    years = clf.predict(test_x)
    print ("Writing.......")
    file = open('result_elastic.csv','w')
    file.write("dataid,prediction\n")
    for i in range(len(years)):
        file.write("{}, {} \n".format(i + 1, years[i]))
    file.close()

if __name__ == "__main__":
    alphas = [0.005,0.01,0.05,0.1,0.2]
    best_alpha = None
    diff = math.inf
    for alp in alphas:
        temp = main(alp)
        if temp < diff:
            diff = temp
            best_alpha = alp
    print("optimal alpha is: "+str(best_alpha))

