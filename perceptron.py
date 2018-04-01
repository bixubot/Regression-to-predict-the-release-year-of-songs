import scipy.io as sio
from sklearn import linear_model
import math

TRAIN_DIR = "data/md_train.mat"
TEST_DIR = "data/md_test.mat"
TEST_SIZE = 10000

def main (alp, ratio):
    train_x, train_y, test_x, test_y = (sio.loadmat(TRAIN_DIR)['trainx'], sio.loadmat(TRAIN_DIR)['trainy'], sio.loadmat(TEST_DIR)['testx'], sio.loadmat(TEST_DIR)['testy'])
    
    clf = linear_model.Lasso()
    clf.fit(train_x,train_y)
    years = clf.predict(test_x)
    
    diff = 0
    for (i, j) in zip(years, test_y):
        diff += abs(i-j)
    diff /= TEST_SIZE   
    print ("MSE is: " + str(diff) +"with alpha: "+str(alp)+", l1_ratio: "+str(ratio))
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
    alphas = [0.01,0.1,1]
    ratios = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    best_alpha, best_ratio = None, None
    diff = math.inf
    for alp in alphas:
        for r in ratios:
            temp = main(alp, r)
            if temp < diff:
                diff = temp
                best_alpha = alp
                best_ratio = r
    print("optimal alpha is: "+str(best_alpha)+ " and l1_ratio is: "+str(best_ratio))

