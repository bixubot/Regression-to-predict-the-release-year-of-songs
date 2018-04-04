import scipy.io as sio
import numpy as np
from sklearn import linear_model, cluster

FULL_DIR = "../MSdata.mat"
TRAIN_DIR = "data/md_train.mat"
TEST_DIR = "data/md_test.mat"
TEST_SIZE = 10000

def kmeans(n_cls):
    train_size = 463715
    test_size = 51630
    train_x, train_y, test_x = (sio.loadmat(FULL_DIR)['trainx'], sio.loadmat(FULL_DIR)['trainy'], sio.loadmat(FULL_DIR)['testx'])
    
    #train_size = 50000
    #test_size = 10000
    #train_x, train_y, test_x, test_y= (sio.loadmat(TRAIN_DIR)['trainx'], sio.loadmat(TRAIN_DIR)['trainy'], sio.loadmat(TEST_DIR)['testx'], sio.loadmat(TEST_DIR)['testy'])


    full_data = np.concatenate((train_x, test_x), axis=0)

    cls = cluster.KMeans(n_clusters=n_cls, n_jobs=4)
    cls.fit(full_data)
    predictions = cls.labels_
    train_clusters = {}
    test_clusters = {}
    for i in range(train_size+test_size):
        label = predictions[i]
        if i < train_size:
            if label not in train_clusters:
                train_clusters[label] = [i]
            else:
                train_clusters[label] += [i]
        else:
            if label not in test_clusters:
                test_clusters[label] = [i-train_size]
            else:
                test_clusters[label] += [i-train_size]
    
    set_train_x = []
    set_train_y = []
    test_x_clusters = predictions[train_size:]

    keys = list(train_clusters.keys())
    for key in keys:
        set_train_x.append(train_x[train_clusters[key], :])
        set_train_y.append(train_y[train_clusters[key], :])
    
    clfs = []
    for i in range(len(set_train_x)):
        clf = ridge(set_train_x[i], set_train_y[i])
        clfs.append(clf)

    years = []
    for j in range(test_size):
        label = test_x_clusters[j]
        years.append(clfs[keys.index(label)].predict(test_x[j, :].reshape(1,-1))[0])

    #evaluate(test_y, years)
    write(years)
    

def ridge(train_x, train_y):
    clf = linear_model.RidgeCV(alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0], normalize=True, cv=10, store_cv_values=False)
    print ("Training......")
    clf.fit(train_x,train_y)
    return clf

def evaluate(test_y, years):
    diff = 0.0
    for i,j in zip(test_y, years):
        diff += abs(i-j)
    diff /= TEST_SIZE
    print("MSE is: "+str(diff))

def predict (clf, test_x):
    years = clf.predict(test_x)
    return years

def write(years):
    print("Writing.......")
    file = open('result_kmeans_ridge.csv','w')
    file.write("dataid,prediction\n")
    for i in range(len(years)):
        file.write("{}, {} \n".format(i + 1, years[i][0]))
    file.close()

if __name__ == "__main__":
    num_cls = [4]
    # 4 is the best among [2,4,6,8,10]
    for n_cls in num_cls:
        kmeans(n_cls)

