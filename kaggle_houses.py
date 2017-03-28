from __future__ import division, print_function

import numpy as np 
import pandas as pd 

from scipy.stats import skew

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor


def load_and_preprocess():

    """
    Load the data (either train.csv or test.csv) and pre-process it with some simple 
    transformations. Return in the correct form for usage in scikit-learn. 

    Arguments
    ---------

    filestr: string
        string pointing to csv file to load into pandas

    Returns
    -------

    X_train: numpy.array
        array containing features of training set

    X_test: numpy.array
        array containing features of test set

    y: numpy.array
        array containing labels for training set

    test_ID: numpy.array
        IDs for test set, for submission

    """

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],\
                          test.loc[:,'MSSubClass':'SaleCondition']))

    #first extract the target variable, and log-transform because the prices are very skewed
    y_train = np.log1p(train['SalePrice'].values) 

    #one hot encoding for categorical variables
    data = pd.get_dummies(data)

    #first find which numerical features are significantly skewed and transform them to log(1 + x)
    numerical = data.dtypes[data.dtypes!='object'].index
    skewed = data[numerical].apply(lambda u: skew(u.dropna()))
    skewed = skewed[skewed > 0.75].index
    data[skewed] = np.log1p(data[skewed])

    #if numerical values are missing, replace with median from that column
    data = data.fillna(data.median())

    X_train = data[:train.shape[0]].as_matrix()
    X_test = data[train.shape[0]:].as_matrix()

    return X_train,X_test,y_train,test.Id



def error(model,X,y):
    """
    Compute the mean absolute error on the data with cross validation

    Arguments
    ---------

    model: object 
        sklearn regressor 

    Returns
    -------

    error: float
        the mean absolute error 

    """

    return np.mean(-cross_val_score(model,X,y,scoring='neg_mean_absolute_error',cv=5))



def fit_lasso(X,y,**kwargs):
    """
    Fit a linear model to the data with L1 regularisation

    Arguments
    ---------

    X: numpy.array
        feature matrix for the training data
    y: numpy.array
        label matrix for the training data

    Returns
    -------

    model: sklearn.linear_model.LassoCV
        the L1 regularised linear model
    """
    #note that alpha ~ 0.0003 seems to work best 
    model = LassoCV(**kwargs)
    model.fit(X,y)
    return model

def fit_randomforest(X,y,**kwargs):

    """
    Fit a random forest regressor to the data

    Arguments
    ---------

    X: numpy.array
        feature matrix for the training data

    y: numpy.array
        label matrix for the training data

    Returns
    -------

    model: sklearn.ensemble.RandomForestRegressor
        random forest regressor

    """

    model = RandomForestRegressor(**kwargs)
    model.fit(X,y)
    return model

def fit_neural_network(X,y,scale=False):

    """
    Fit a simple neural network with one hidden layer using keras and the scikit-learn wrapper.

    Arguments
    ---------

    X: numpy.array
        feature matrix for the training data

    y: numpy.array
        label matrix for the training data

    scale: bool
        if True, use sklean RobustScaler to scale the 
        features before training

    Returns
    -------

    model: KerasRegressor
        neural network

    X_train: numpy.array
        feature matrix for the training data

    y_train: numpy.array
        label matrix for the training data

    X_val: numpy.array
        feature matrix for the validation data

    y_val: numpy.array
        label matrix for the validation data
    """

    if scale:
        X = RobustScaler().fit_transform(X)

    X_train,X_val,y_train,y_val = train_test_split(X,y,random_state=0)

    def build_model(): 
        model = Sequential()
        model.add(Dense(13,input_dim=X.shape[1],kernel_initializer='normal',activation='relu'))
        model.add(Dense(6,input_dim=X.shape[1],kernel_initializer='normal',activation='relu'))
        model.add(Dense(1,kernel_initializer='normal'))
        model.compile(loss='mean_absolute_error',optimizer='adam')
        return model

    model = KerasRegressor(build_fn=build_model,epochs=100,batch_size=30)
    model.fit(X_train,y_train)

    return model,X_train,y_train,X_val,y_val

def model_bagging_predictions(X_train,X_test,y_train,test_IDs,make_submission=True):

    """
    Use lasso, RF and NN models at once and average the results to obtain
    (hopefully) less biased predictions.

    Arguments
    ---------

    X_train: numpy.array
        feature matrix for the training data

    X_test: numpy.array
        feature matrix for the test data

    y_train: numpy.array
        label matrix for the training data

    Returns
    -------

    y_pred: numpy.array
        averaged prediction from all three regressors

    y_lasso: numpy.array
        lasso prediction

    y_forest: numpy.array
        random forest prediction

    y_neural_net: numpy.array
        neural net prediction 

    submission: pandas.DataFrame
        (if make_submission is True)
        dataframe with two columns, ID and predicted sale price for test set
    """


    lasso = fit_lasso(X_train,y_train)
    forest = fit_randomforest(X_train,y_train)


    scaler = RobustScaler()
    X_transformed = scaler.fit_transform( np.vstack((X_train,X_test)) )
    X_train_transformed = X_transformed[:X_train.shape[0]]
    X_test_transformed = X_transformed[X_train.shape[0]:] 
    neural_net,_,__,___,____ = fit_neural_network(X_train_transformed,y_train)

    y_lasso = lasso.predict(X_test)
    y_forest = forest.predict(X_test)
    y_neural_net = neural_net.predict(X_test_transformed)

    y_pred = (1./3.)*(y_lasso+y_forest+y_neural_net)

    if make_submission:
        predicted_prices = np.exp(y_pred)-1.
        submission = pd.DataFrame({'id':test_IDs,'SalePrice':predicted_prices})
        return y_pred,y_lasso,y_forest,y_neural_net,submission
    else:
        return y_pred,y_lasso,y_forest,y_neural_net





