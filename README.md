## Regression-to-predict-the-release-year-of-songs
The project is assigned for COMS 4771 Spring 2018 at the Columbia University. The goal for this assignment is to develop a good quality regressor for the given dataset.  The given dataset comes from musical songs domain. It consists of about 1/2 Million songs from 20th and 21st century. The task is to predict the year the song was released given the musical content of the song.

1. parser.py: takes the original dataset and shrinks it down to a workable size to be able to train and test efficiently and be stored on the Github.
2. data: contains two parsed datasets, "sm_train.mat" and "sm_test.mat". "sm_train.mat" contains two variables "trainx" and "trainy" with 5000 sample, each of which contains 90 features. And "sm_test.mat" contains variables "testx" and "testy", with 2000 samples in the same shape.


