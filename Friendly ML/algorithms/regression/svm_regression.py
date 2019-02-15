# SVR

from time import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import expon, reciprocal
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

def svm_regression(features, label, auto):
    start=time()
    
    label = label.values
    
    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size = 0.2)
    
    # Feature Scaling
    sc_X = StandardScaler()
    sc_y = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    y_train = y_train.reshape(-1, 1)
    y_train = sc_y.fit_transform(y_train)
    
    # Building regressor
    regressor = SVR()
    
    if auto == 'y':
        # Random Search
        param_distribs = {"C": reciprocal(20, 200000),
                          'kernel': ['linear', 'rbf'],
                          "gamma": expon(scale=1.0)}
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
    return {'name':'SVM Regression','model':regressor, 'train_rmse':train_rmse, 'test_rmse':test_rmse, 'duration':(time()-start)/60}