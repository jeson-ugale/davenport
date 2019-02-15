# XGBoost Classification

from time import time
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from scipy.stats import randint as sp_randint
from scipy.stats import uniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score, StratifiedKFold

def xgb_classification(features, label, auto):
    start=time()
    
    features = features.values
    label = label.values
    
    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size = 0.2)
    
    # Fitting XGBoost to the Training set
    classifier = XGBClassifier()
    
    if auto == 'y':
        # Random Search
        param_distribs = {"learning_rate": uniform(0.01,0.09), # 0.01 - 0.1
                          "n_estimators": sp_randint(100,1000),
                          "max_depth": sp_randint(3, 10), #3-10
                          "subsample": uniform(0.8,0.2), #0.8-1
                          "colsample_bytree": uniform(0.3, 0.7), #0.3-1
                          "gamma": sp_randint(0, 5)} #0-5
        rnd_search = RandomizedSearchCV(classifier, param_distributions=param_distribs,
                                        n_iter=60, cv=10, scoring='accuracy',
                                        verbose=2, n_jobs=1)
        rnd_search.fit(X_train, y_train)
        best_accuracy = rnd_search.best_score_
        classifier = rnd_search.best_estimator_
    else:
        # Applying k-Fold Cross Validation
        kfold = StratifiedKFold(n_splits=10)
        accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = kfold)
        best_accuracy = accuracies.mean()
    
    # Predicting the Test set results
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    unique_label = np.unique(y_test)
    cm = pd.DataFrame(confusion_matrix(y_test, y_pred, labels=unique_label), 
                       index=['true:{:}'.format(x) for x in unique_label], 
                       columns=['pred:{:}'.format(x) for x in unique_label])
    return {'name':'XGBoost Classification', 'model':classifier, 'train_accuracy':best_accuracy, 'cm':cm, 'duration':(time()-start)/60}