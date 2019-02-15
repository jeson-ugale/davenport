# Naive Bayes

from time import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix

def naive_bayes(features, label):
    start=time()
    
    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size = 0.2)
    
    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)    
    
    # Fitting Naive Bayes to the Training set
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    # Applying k-Fold Cross Validation
    kfold = StratifiedKFold(n_splits=10)
    accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = kfold)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    unique_label = np.unique(y_test)
    cm = pd.DataFrame(confusion_matrix(y_test, y_pred, labels=unique_label), 
                      index=['true:{:}'.format(x) for x in unique_label], 
                      columns=['pred:{:}'.format(x) for x in unique_label])
    return {'name':'Naive Bayes', 'model':classifier, 'train_accuracy':accuracies.mean(), 'cm':cm, 'duration':(time()-start)/60}