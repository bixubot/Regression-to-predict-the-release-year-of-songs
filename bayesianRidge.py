import scipy.io as sio
import numpy as np
from sklearn import linear_model
import math

TRAIN_DIR = "data/md_train.mat"
TEST_DIR = "data/md_test.mat"
TEST_SIZE = 10000

def main (alp1, alp2, lbd1, lbd2):
    train_x, train_y, test_x, test_y = (sio.loadmat(TRAIN_DIR)['trainx'], sio.loadmat(TRAIN_DIR)['trainy'], sio.loadmat(TEST_DIR)['testx'], sio.loadmat(TEST_DIR)['testy'])
    train_y = train_y.ravel()
    test_y = test_y.ravel()
    clf = linear_model.BayesianRidge(alpha_1=alp1, alpha_2=alp2, lambda_1=lbd1, lambda_2=lbd2)
    clf.fit(train_x,train_y)
    years = clf.predict(test_x)
    
    diff = float(0.0)
    for (i, j) in zip(years, test_y):
        diff += abs(i-j)
    diff /= TEST_SIZE   
    print ("MSE is: " + str(diff) +" with alpha: "+str(alp1)+' '+str(alp2)+' '+str(lbd1)+' '+str(lbd2))
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
    # optimal: 0.1, 0.0001, 0.0001, 0.1

    # 5^4 comnbinations, run with caution
    alpha1s = [0.0001,0.001,0.005,0.01,0.05,0.1]
    alpha2s = alpha1s
    lambda1s = alpha1s
    lambda2s = alpha1s
    best_alpha = None
    diff = math.inf
    for alp1 in alpha1s:
        for alp2 in alpha2s:
            for lbd1 in lambda1s:
                for lbd2 in lambda2s:
                    temp = main(alp1,alp2,lbd1,lbd2)
                    if temp < diff:
                        diff = temp
                        best_alpha = [alp1,alp2,lbd1,lbd2]
    print("optimal alpha is: ")
    print(best_alpha)

