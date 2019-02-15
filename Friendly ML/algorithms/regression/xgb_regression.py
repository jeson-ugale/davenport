# XGBoost Regression

from time import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
from scipy.stats import uniform
from scipy.stats import randint as sp_randint
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV

def xgb_regression(features, label, auto):
    start=time()
    
    features = features.values
    label = label.values
    
    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size = 0.2)
    
    # Fitting XGBoost to the Training set
    regressor = XGBRegressor()
    
    if auto == 'y':
        # Random Search
        param_distribs = {"learning_rate": uniform(0.01,0.09), # 0.01 - 0.1
                          "n_estimators": sp_randint(100,1000),
                          "max_depth": sp_randint(3, 10), #3-10
                          "subsample": uniform(0.8,0.2), #0.8-1
                          "colsample_bytree": uniform(0.3, 0.7), #0.3-1
                          "gamma": sp_randint(0, 5)} #0-5
        rnd_search = RandomizedSearchCV(regressor, param_distributions=param_distribs,
                                        n_iter=60, cv=10, scoring='neg_mean_squared_error',
                                        verbose=2, n_jobs=1)
        rnd_search.fit(X_train, y_train)
        train_mse = rnd_search.best_score_
        train_rmse = np.sqrt(-train_mse)
        regressor = rnd_search.best_estimator_
    else:
        regressor.fit(X_train, y_train)
        # Applying k-Fold Cross Validation
        accuracies = cross_val_score(estimator = regressor, X = X_train, y = y_train, cv = 10, scoring='neg_mean_squared_error')
        train_mse = accuracies.mean()
        train_rmse = np.sqrt(-train_mse)
    
    # Predicting the Test set results
    y_pred = regressor.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return {'name':'XGBoost Regression','model':regressor, 'train_rmse':train_rmse, 'test_rmse':test_rmse, 'duration':(time()-start)/60}