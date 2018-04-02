import scipy.io as sio
from sklearn import linear_model, ensemble
import math

TRAIN_DIR = "data/md_train.mat"
TEST_DIR = "data/md_test.mat"
TEST_SIZE = 10000

def main (n_trees, n_fea):
    train_x, train_y, test_x, test_y = (sio.loadmat(TRAIN_DIR)['trainx'], sio.loadmat(TRAIN_DIR)['trainy'], sio.loadmat(TEST_DIR)['testx'], sio.loadmat(TEST_DIR)['testy'])
    train_y = train_y.ravel()
    test_y = test_y.ravel()
     
    clf = ensemble.RandomForestRegressor(n_estimators=n_trees, n_jobs=4, min_impurity_split=0.5, max_features=n_fea)
    clf.fit(train_x,train_y)
    years = clf.predict(test_x)
    
    diff = 0
    for (i, j) in zip(years, test_y):
        diff += abs(i-j)
    diff /= TEST_SIZE   
    print ("MSE is: " + str(diff) +"with n_trees: "+str(n_trees)+" and n_features is: "+str(n_fea))
    
    return diff, clf
    
def predict(clf):
    train_x, train_y, test_x = (sio.loadmat('../MSdata.mat')['trainx'], sio.loadmat('../MSdata.mat')['trainy'], sio.loadmat('../MSdata.mat')['testx'])
    train_y = train_y.ravel()

    clf.fit(train_x, train_y)
    years = clf.predict(test_x)

    print ("Writing.......")
    file = open('result_randomForest.csv','w')
    file.write("dataid,prediction\n")
    for i in range(len(years)):
        file.write("{}, {} \n".format(i + 1, years[i]))
    file.close()

if __name__ == "__main__":
    forests = [20,50,100,200,250]
    n_feas = [40,50,60]
    best_n, best_fea, best_clf = None, None, None
    diff = math.inf
    for n_trees in forests:
        for n_fea in n_feas:
            temp, clf = main(n_trees, n_fea)
            if temp < diff:
                diff = temp
                best_n = n_trees
                best_fea = n_fea
                best_clf = clf
    print("optimal number of trees is: "+str(best_n)+" and n_features is: "+str(best_fea))
    #predict(best_clf)
