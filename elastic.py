import scipy.io as sio
from sklearn import linear_model

def main ():
    train_x, train_y, test_x, test_y = (sio.loadmat('data/sm_train.mat')['trainx'], sio.loadmat('data/sm_train.mat')['trainy'], sio.loadmat('data/sm_test.mat')['testx'], sio.loadmat('data/sm_test.mat')['testy'])

    clf = linear_model.RidgeCV(alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0], normalize=True, cv=10, store_cv_values=False)
    print ("Training......")
    clf.fit(train_x,train_y)
    print ("Predicting.......")
    years = clf.predict(test_x)
    print("Writing.......")
    file = open('result_ridge.csv','w')
    file.write("dataid,prediction\n")
    for i in range(len(years)):
        file.write("{}, {} \n".format(i + 1, years[i][0]))
    file.close()

if __name__ == "__main__":
	main()

