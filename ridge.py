import scipy.io as sio
from sklearn import linear_model

TRAIN_DIR = "data/md_train.mat"
TEST_DIR = "data/md_test.mat"
TEST_SIZE = 10000

def main ():
    train_x, train_y, test_x, test_y = (sio.loadmat(TRAIN_DIR)['trainx'], sio.loadmat(TRAIN_DIR)['trainy'], sio.loadmat(TEST_DIR)['testx'], sio.loadmat(TEST_DIR)['testy'])
    clf = linear_model.RidgeCV(alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0], normalize=True, cv=10, store_cv_values=False)
    print ("Training......")
    clf.fit(train_x,train_y)
    print ("Predicting.......")
    years = clf.predict(test_x)

    diff = 0.0
    for i,j in zip(test_y, years):
        diff += abs(i-j)
    diff /= TEST_SIZE
    print("MSE is: "+str(diff))

    # predict(clf)
    

def predict (clf):
    train_x, train_y, test_x = (sio.loadmat("../MSdata.mat")["trainx"], sio.loadmat("../MSdata.mat")['trainy'], sio.loadmat('../MSdata.mat')['testx'])

    clf.fit(train_x, train_y)
    years = clf.predict(test_x)

    print("Writing.......")
    file = open('result_ridge.csv','w')
    file.write("dataid,prediction\n")
    for i in range(len(years)):
        file.write("{}, {} \n".format(i + 1, years[i][0]))
    file.close()

if __name__ == "__main__":
	main()

