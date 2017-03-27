from __future__ import division, print_function

import numpy as np 
import pandas as pd 

from scipy.stats import skew
from sklearn.preprocessing import robust_scale
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

def load_and_preprocess(filestr="data/train.csv"):

    """
    Load the data (either train.csv or test.csv) and pre-process it with some simple 
    transformations. Return in the correct form for usage in scikit-learn. 

    Arguments
    ---------

    filestr: string
        string pointing to csv file to load into pandas

    Returns
    -------

    X: numpy.array
        array containing features

    y: numpy.array
        array containing target values

    """

    data = pd.read_csv(filestr)

    #first extract the target variable, and log-transform because the prices are very skewed
    y = np.log1p(data['SalePrice'].values) 
    data = data.loc[:,'MSSubClass':'SaleCondition']

    #one hot encoding for categorical variables
    data = pd.get_dummies(data)

    #first find which numerical features are significantly skewed and transform them to log(1 + x)
    numerical = data.dtypes[data.dtypes!='object'].index
    skewed = data[numerical].apply(lambda u: skew(u.dropna()))
    skewed = skewed[skewed > 0.75].index
    data[skewed] = np.log1p(data[skewed])

    #if numerical values are missing, replace with median from that column
    data = data.fillna(data.median())

    X = data.as_matrix()

    return X,y


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

    parameter_grid = grid_forest = {
                                    'n_estimators': [50,100,150]
                                    }

    model = RandomForestRegressor(**kwargs)
    grid_search = GridSearchCV(model,parameter_grid,n_jobs=3)
    grid_search.fit(X,y)
    return grid_search.best_estimator_





