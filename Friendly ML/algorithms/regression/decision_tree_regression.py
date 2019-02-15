# Decision Tree Regression

from time import time
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

def decision_tree_regression(features, label, auto):
    start=time()
    
    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size = 0.2)
    
    # Fitting Decision Tree Regression to the dataset
    regressor = DecisionTreeRegressor()
    
    if auto == 'y':
        # Random Search
        param_distribs = {"criterion": ["mse", "mae"],
                          "min_samples_split": sp_randint(2, 11)}
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
    return {'name':'Decision Tree Regression','model':regressor, 'train_rmse':train_rmse, 'test_rmse':test_rmse, 'duration':(time()-start)/60}