# K-Nearest Neighbors Classification

from time import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold

def knn_classification(features, label, auto):
    start = time()
    
    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size = 0.2)
    
    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)    
    
    # Building classifier
    classifier = KNeighborsClassifier()

    if auto == 'y':
        # Random Search
        param_distribs = {"n_neighbors": sp_randint(3, 10),
                          "weights": ['uniform', 'distance']}
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
    return {'name':'KNN Classification', 'model':classifier, 'train_accuracy':best_accuracy, 'cm':cm, 'duration':(time()-start)/60}