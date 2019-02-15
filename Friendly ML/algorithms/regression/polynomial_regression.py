# Polynomial Regression

from time import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.metrics import mean_squared_error

def polynomial_regression(features, label):
    start=time()
    
    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size = 0.2)
    
    # Fitting Polynomial Regression to the dataset
    poly_reg = PolynomialFeatures()
    X_train = poly_reg.fit_transform(X_train)
    X_test = poly_reg.fit_transform(X_test)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    
    # Applying k-Fold Cross Validation
    accuracies = cross_val_score(estimator = regressor, X = X_train, y = y_train, cv = 10, scoring='neg_mean_squared_error')
    train_mse = accuracies.mean()
    train_rmse = np.sqrt(-train_mse)
    
    # Predicting the Test set results
    y_pred = regressor.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return {'name':'Polynomial Regression','model':regressor, 'train_rmse':train_rmse, 'test_rmse':test_rmse, 'duration':(time()-start)/60}