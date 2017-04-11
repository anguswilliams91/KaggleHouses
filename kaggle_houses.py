from __future__ import division, print_function

import numpy as np 
import pandas as pd 

from scipy.stats import skew

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LassoCV,RidgeCV
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor 
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers.core import Dropout

#first attempt at a kaggle competition, experiment with different regressors and follow some of the advice in 
#kernels.

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
    data = data.fillna(data.mean())

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

def fit_ridge(X,y,**kwargs):
    """
    Fit a linear model to the data with L2 regularisation

    Arguments
    ---------

    X: numpy.array
        feature matrix for the training data
    y: numpy.array
        label matrix for the training data

    Returns
    -------

    model: sklearn.linear_model.RidgeCV
        the L2 regularised linear model
    """
    #note that alpha ~ 0.0003 seems to work best 
    model = RidgeCV(**kwargs)
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
    """

    if scale:
        X = RobustScaler().fit_transform(X)

    #X_train,X_val,y_train,y_val = train_test_split(X,y,random_state=0)

    def build_model(): 
        model = Sequential()
        model.add(Dense(500,input_dim=X.shape[1],kernel_initializer='normal',activation='relu'))
        model.add(Dropout(0.05))
        model.add(Dense(1,kernel_initializer='normal'))
        model.compile(loss='mean_absolute_error',optimizer='adam')
        return model

    model = KerasRegressor(build_fn=build_model,epochs=200,batch_size=100)
    model.fit(X,y,validation_split=0.25)

    return model#,X_train,y_train,X_val,y_val

def fit_gradient_boost(X,y,**kwargs):

    """
    Fit a gradient boosting regressor to the data

    Arguments
    ---------

    X: numpy.array
        feature matrix for the training data

    y: numpy.array
        label matrix for the training data

    Returns
    -------

    model: sklearn.ensemble.GradientBoostingRegressor
        Gradient boosting regressor

    """

    model = GradientBoostingRegressor(**kwargs)
    model.fit(X,y)

    return model

def fit_SVR(X,y,scale=True,**kwargs):

    """
    Fit a Support Vector Regressor to the data

    Arguments
    ---------

    X: numpy.array
        feature matrix for the training data

    y: numpy.array
        label matrix for the training data

    Returns
    -------

    model: sklearn.svr.SVR
        Best performing SVR based on cross validation

    """

    if scale: Xs = RobustScaler().fit_transform(X)
    params = {'kernel':['rbf','linear','poly'],'C':[0.001,0.01,0.1,1.]}
    model = GridSearchCV(SVR(),params,n_jobs=6,verbose=10)
    model.fit(Xs,y)

    return model

def fit_extra_trees(X,y,**kwargs):

    """
    Fit a extra trees regressor to the data

    Arguments
    ---------

    X: numpy.array
        feature matrix for the training data

    y: numpy.array
        label matrix for the training data

    Returns
    -------

    model: sklearn.ensemble.ExtraTreesRegressor
        random forest regressor

    """

    model = ExtraTreesRegressor(**kwargs)
    model.fit(X,y)
    return model


def model_bagging_predictions(X_train,X_test,y_train,test_IDs):

    """
    Fit a bunch of estimators, then fit a meta-estimator to them.

    Arguments
    ---------

    X_train: numpy.array
        array containing features of training set

    X_test: numpy.array
        array containing features of test set

    y: numpy.array
        array containing labels for training set

    test_IDs: numpy.array
        IDs for test set, for submission

    Returns
    -------

    submission: pandas.DataFrame
        A pandas dataframe for submitting

    """

    X1,X2,y1,y2 = train_test_split(X_train,y_train) 
    rs = RobustScaler()
    X1s = rs.fit_transform(X1)
    X2s = rs.transform(X2)
    lasso = fit_lasso(X1,y1,alphas=np.array([0.0003]))
    rf = fit_randomforest(X1,y1,n_estimators=200)
    gb = fit_gradient_boost(X1,y1,n_estimators=200)
    ridge = fit_ridge(X1,y1,alphas=np.linspace(1.,20.,19))
    svr = fit_SVR(X1s,y1)
    et = fit_extra_trees(X1,y1,n_estimators=200)
    models = {'lasso':lasso, 'rf':rf, 'gb':gb, 'svr':svr, 'ridge':ridge, 'et':et}
    predictions = np.zeros((X2.shape[0],len(models.keys())))
    for i,name in enumerate(models.keys()):
        print(name)
        if name != 'svr':
            predictions[:,i] = models[name].predict(X2)
        else:
            predictions[:,i] = models[name].predict(X2s)

    X2_meta = np.concatenate((X2,predictions),1)
    lasso_meta = fit_lasso(X2_meta,y2,alphas=np.linspace(0.0005,0.005,10))

    #now predict for the training set
    Xts = rs.transform(X_test)
    predictions_test = np.zeros((X_test.shape[0],len(models.keys())))
    for i,name in enumerate(models.keys()):
        print(name)
        if name != 'svr':
            predictions_test[:,i] = models[name].predict(X_test)
        else:
            predictions_test[:,i] = models[name].predict(Xts)

    X_test_meta = np.concatenate((X_test,predictions_test),1)
    y_pred = lasso_meta.predict(X_test_meta)

    return pd.DataFrame({'Id':test_IDs, 'SalePrice': np.exp(y_pred)-1.})       

def main():
    np.random.seed(20)
    X_train,X_test,y_train,ids = load_and_preprocess()
    submission = model_bagging_predictions(X_train,X_test,y_train,ids)
    submission.to_csv("data/submission.csv",index=False)


if __name__ == "__main__":
    main()





