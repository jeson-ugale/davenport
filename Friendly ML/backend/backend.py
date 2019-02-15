# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 12:12:37 2019

@author: jeson
"""

# Importing the libraries
import numpy as np
import pandas as pd

# Request learning type
def request_learning():
    print("-"*40)
    learning_type = input('Type of learning:' +
    '\n\n 1. supervised' +
    '\n 2. unsupervised' + 
    '\n\n Enter the option number: ')
    return learning_type

# Request supervised/unsupervised problem
def request_problem(learning_type):
    print("-"*40)
    if (learning_type == '1'):
        problem = input('Type of supervised learning problem:' +
        '\n\n 1. regression' +
        '\n 2. classification' + 
        '\n\n Enter the option number: ')
    elif (learning_type == '2'):
        problem = input('Type of unsupervised learning problem:' +
        '\n\n 1. clustering' +
        '\n 2. association rule learning' + 
        '\n\n Enter the option number: ')
    if (learning_type == '1'):
        if (problem == '1'):
            problem = 'regression'
        elif (problem == '2'):
            problem = 'classification'
    elif (learning_type == '2'):
        if (problem == '1'):
            problem = 'clustering'
        elif (problem == '2'):
            problem = 'association'
    return problem

# Request dataset
def request_dataset(problem):
    print("-"*40)
    import os
    print('\nFiles currently in datasets folder:\n')
    for i in range(0, len(os.listdir('datasets'))):
        print(str(i+1) + '. ' + os.listdir('datasets')[i])
    if (problem == 'association'):
        dataset = pd.read_csv('datasets/' + os.listdir('datasets')[int(input('Name of .csv: '))-1], header = None)
    else:
        dataset = pd.read_csv('datasets/' + os.listdir('datasets')[int(input('Name of .csv: '))-1])
        dataset = dataset.dropna()
    print("\nRows x Columns")
    print(dataset.shape)
    print("\n")
    return dataset

# Request label
def request_label(dataset):
    print("-"*40)
    print("\nPotential Labels\n")
    count = 0
    for column in dataset.columns:
        print(str(count) + ". " +  column)
        count += 1
    label = int(input('Index of dependent variable: '))
    y = dataset.iloc[:, label]
    dataset = dataset.drop(dataset.columns[label], axis=1)
    #y = y.values
    return [y, dataset]

def remove_features(dataset):
    print("-"*40)
    print("\nWant to remove any features?\n")
    count = 0
    for column in dataset.columns:
        print(str(count) + ". " +  column)
        count += 1
    unwanted = input('Indices of unwanted features: ')
    if unwanted == 'none':
        return dataset
    unwanted = [int(x) for x in unwanted.split(',')]
    subset = dataset.drop(dataset.columns[unwanted], axis=1)
    return subset

# Encode features
def encode_features(dataset):
    one_hot = dataset.select_dtypes(exclude=[np.number])
    ordinal = dataset.select_dtypes(exclude=[np.number])
    if len(one_hot.columns)==0:
        return dataset
    dataset = dataset.select_dtypes([np.number])
    print("-"*40)
    print("\nCategorical Variables\n")
    count = 0
    for column in one_hot.columns:
        print(str(count) + ". " + column)
        count += 1

    # Ordinal Encoding
    to_encode = input('Indices of categoricals with order: ')
    if (to_encode == 'none'):
        pass
    else:
        to_encode = [int(x) for x in to_encode.split(',')]
        one_hot = one_hot.drop(one_hot.columns[to_encode], axis=1)
        #ordinal = ordinal.iloc[:, to_encode]
        for i in to_encode:
            print(ordinal.columns[i])
            count = 0
            unique = ordinal.iloc[:,i].unique()
            for uniq in unique:
                print(str(count) + ". " + uniq)
                count += 1
            order = input('Order options from smallest to largest: ')
            order = [int(x) for x in order.split(',')]
            unique = [unique[i] for i in order]
            ordinal.iloc[:,i] = ordinal.iloc[:,i].map({b:a for a, b in enumerate(unique)})
            dataset = pd.concat([ordinal.iloc[:,i], dataset], axis=1)
    
    # One-Hot Encoding
    for column in one_hot:
        dummies = pd.get_dummies(one_hot[column], prefix=column)
        dummies = dummies.iloc[:, 1:]
        dataset = pd.concat([dummies, dataset], axis=1)
        
    return dataset

def request_features(dataset, problem, y=None):
    print("-"*40)
    if problem == "clustering":
        print("\nPotential Features\n")
        count = 0
        for column in dataset.columns:
            print(str(count) + ". " + column)
            count += 1
        features = input('Indices of features: ')
        if (features == '*'):
            return dataset
        features = [int(x) for x in features.split(',')]
        X = dataset.iloc[:, features]
        return X
    X_temp = dataset.copy()
    y_temp = y.copy()
    # Put indices beside column names
    count = 0
    for column in X_temp.columns:
        X_temp.rename({column:str(count) + ". " + column}, axis=1, inplace=True)
        count += 1
    if problem == "regression":
        from sklearn.ensemble import ExtraTreesRegressor
        model = ExtraTreesRegressor()
    elif problem == "classification":
        from sklearn.ensemble import ExtraTreesClassifier
        model = ExtraTreesClassifier()
    model.fit(X_temp, y_temp)
    imp_feat = model.feature_importances_
    feature_importance = dict(zip(X_temp.columns, imp_feat))
    data_sorted = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    print("\nFeature Importances\n")
    for tup in data_sorted:
        print("- " + tup[0] + ": " + str(tup[1]))
    features = input('Indices of features: ')
    if (features == '*'):
        return dataset
    features = [int(x) for x in features.split(',')]
    X = dataset.iloc[:, features]
    return X

# Request dataset/features/label
def request_all_data(problem):
    data_collection = {}
    data_collection['dataset'] = request_dataset(problem)
    
    # If association rule learning
    if (problem == 'association'):
        return data_collection
    
    # If regression/classification
    if (problem == 'regression' or problem == 'classification'):
        updated_data = request_label(data_collection['dataset'])
        data_collection['label'] = updated_data[0]
        data_collection['dataset'] = updated_data[1]

    # If clustering/regression/classification
    data_collection['dataset'] = remove_features(data_collection['dataset'])
    data_collection['dataset'] = encode_features(data_collection['dataset'])
    if (problem == 'clustering'):
        data_collection['features'] = request_features(data_collection['dataset'], problem)
    else:
        data_collection['features'] = request_features(data_collection['dataset'], problem, data_collection['label'])
    
    return data_collection
    
# Request Model
def request_model(problem, data):
    if (problem == 'regression'):
        results = request_regression(data)
        if len(results)==1:
            print('-'*40)
            print("Here's the optimal %s model:\n" % results[0]['name'])
            print(results[0]['model'])
            print("\nAfter training on your data, this model achieved an RMSE of %.2f." % results[0]['train_rmse'])
            print("On the test/unseen data, this model achieved an RMSE of %.2f." % results[0]['test_rmse'])
            print("It took %.2f minutes to find this model." % results[0]['duration'])
        else:
            count=0
            for model in results:
                print('-'*40)
                print(str(count+1) + ".")
                print("Here's the optimal %s model:\n" % results[count]['name'])
                print(results[count]['model'])
                print("\nAfter training on your data, this model achieved an RMSE of %.2f." % results[count]['train_rmse'])
                print("On the test/unseen data, this model achieved an RMSE of %.2f." % results[count]['test_rmse'])
                print("It took %.2f minutes to find this model." % results[count]['duration'])
                count += 1
            model_choice = int(input("Enter the number of your chosen model: "))
            return [results[model_choice-1]]
        # Here is the model, predict
    elif (problem == 'classification'):
        results = request_classification(data)
        if len(results)==1:
            print('-'*40)
            print("Here's the optimal %s model:\n" % results[0]['name'])
            print(results[0]['model'])
            print("\nAfter training on your data, this model achieved an accuracy of %.2f." % results[0]['train_accuracy'])
            print("On the test/unseen data, this model achieved the following confusion matrix:\n")
            print(results[0]['cm'])
            print("\nIt took %.2f minutes to find this model." % results[0]['duration'])
        else:
            count=0
            for model in results:
                print('-'*40)
                print(str(count+1) + ".")
                print("Here's the optimal %s model:\n" % results[count]['name'])
                print(results[count]['model'])
                print("\nAfter training on your data, this model achieved an accuracy of %.2f." % results[count]['train_accuracy'])
                print("On the test/unseen data, this model achieved the following confusion matrix:\n")
                print(results[0]['cm'])
                print("\nIt took %.2f minutes to find this model." % results[count]['duration'])
                count += 1
            model_choice = int(input("Enter the number of your chosen model: "))
            return [results[model_choice-1]]
        # Here is the model, predict
    elif (problem == 'clustering'):
        results = request_clustering(data)
        if len(results)==1:
            print('-'*40)
            print("Here's the optimal %s model with %d features:\n" % (results[0]['name'], results[0]['opt_clusters']))
            print(results[0]['model'])
            print("\nHere's a preview of the original dataset concatenated with the predicted groupings:")
            print(pd.concat([data['dataset'], results[0]['predictions']], axis=1).head())
            print("\nIt took %.2f minutes to find this model." % results[0]['duration'])
        # Here is the original dataset, with the predicted groupings
    elif (problem == 'association'):
        results = request_association(data)
        if len(results)==1:
            print('-'*40)
            print("Here's a preview of the optimal %s ruleset:\n" % results[0]['name'])
            count = 0
            for i in range(0, 10):
                print(str(count+1) + ". " + str(results[0]['rules'][count][0]) + "  Support: " + "%.4f" % results[0]['rules'][count][1])
                count += 1
            print("\nIt took %.2f minutes to find this ruleset." % results[0]['duration'])
    return results

def predict_until_quit(data, model, problem):
    if problem == 'association':
        return
    to_predict = None
    while to_predict != 'quit':
        print('-'*40)
        if problem == "clustering":
            print("Your %s model takes in the %d following features:\n " % (model[0]['name'], len(data['features'].columns)))
        else:
            print("Your %s model predicts the '%s' label.\n\nIt takes in the %d following features:\n " % (model[0]['name'], data['label'].name, len(data['features'].columns)))
        count = 1
        for feature in data['features'].columns:
            print(str(count) + ". " + feature)
            count+=1
        to_predict = input('Predict the label for a new observation: ')
        if to_predict == 'quit':
            return
        else:
            to_predict = [[float(x) for x in to_predict.split(',')]]
            if problem == "clustering":
                print("\nYour model predicted the following cluster:\n")     
            else:
                print("\nYour model predicted the following '%s' label:\n" % data['label'].name)
            print(model[0]['model'].predict(to_predict))

###############################################################################
# Request algorithm - Regression
            
from algorithms.regression.linear_regression import linear_regression
from algorithms.regression.polynomial_regression import polynomial_regression
from algorithms.regression.svm_regression import svm_regression
from algorithms.regression.decision_tree_regression import decision_tree_regression
from algorithms.regression.random_forest_regression import random_forest_regression
from algorithms.regression.xgb_regression import xgb_regression

def request_regression(data):
    X = data['features']
    y = data['label']
    regression_alg = input('Select regression algorithm:' +
        '\n\n 1. linear regression' +
        '\n 2. polynomial regression' +
        '\n 3. support vector regression' +
        '\n 4. decision tree regression' +
        '\n 5. random forest regression' +
        '\n 6. xgboost regression' +
        '\n\n Enter the option number: ')
    auto = input("Automatic hyperparameter tuning? [y/n]: ")
    if (regression_alg == '1'):
        return [linear_regression(X, y)]
    elif (regression_alg == '2'):
        return [polynomial_regression(X, y)]
    elif (regression_alg == '3'):
        return [svm_regression(X, y, auto)]
    elif (regression_alg == '4'):
        return [decision_tree_regression(X, y, auto)]
    elif (regression_alg == '5'):
        return [random_forest_regression(X, y, auto)]
    elif (regression_alg == '6'):
        return [xgb_regression(X, y, auto)]
    elif (regression_alg == '*'):
        models = []
        models.append(linear_regression(X, y))
        models.append(polynomial_regression(X, y))
        if auto == 'n':
            models.append(svm_regression(X, y, auto))
        models.append(decision_tree_regression(X, y, auto))
        models.append(random_forest_regression(X, y, auto))
        if auto == 'n':
            models.append(xgb_regression(X, y, auto))
        
        return models

###############################################################################
# Request algorithm - Classification
        
from algorithms.classification.logistic_regression import logistic_regression
from algorithms.classification.knn_classification import knn_classification
from algorithms.classification.svm_classification import svm_classification
from algorithms.classification.naive_bayes import naive_bayes
from algorithms.classification.decision_tree_classification import decision_tree_classification
from algorithms.classification.random_forest_classification import random_forest_classification
from algorithms.classification.xgb_classification import xgb_classification

def request_classification(data):    
    X = data['features']
    y = data['label']   
    classification_alg = input('Select classification algorithm:' +            
        '\n\n 1. logistic regression' +
        '\n 2. k-nearest neighbors' +
        '\n 3. svm classification' +
        '\n 4. naive bayes' +
        '\n 5. decision tree classification' +
        '\n 6. random forest classification' +
        '\n 7. xgboost classification' +
        '\n\n Enter the option number: ')
    auto = input("Automatic hyperparameter tuning? [y/n]: ")
    if (classification_alg == '1'):
        return [logistic_regression(X, y, auto)]
    elif (classification_alg == '2'):
        return [knn_classification(X, y, auto)]
    elif (classification_alg == '3'):
        return [svm_classification(X, y, auto)]
    elif (classification_alg == '4'):
        return [naive_bayes(X, y)]
    elif (classification_alg == '5'):
        return [decision_tree_classification(X, y, auto)]
    elif (classification_alg == '6'):
        return [random_forest_classification(X, y, auto)]
    elif (classification_alg == '7'):
        return [xgb_classification(X, y, auto)]
    elif (classification_alg == '*'):
        models = []
        models.append(logistic_regression(X, y, auto))
        models.append(knn_classification(X, y, auto))
        models.append(svm_classification(X, y, auto))
        models.append(naive_bayes(X, y))
        models.append(decision_tree_classification(X, y, auto))
        models.append(random_forest_classification(X, y, auto))
        models.append(xgb_classification(X, y, auto))
        return models
        
###############################################################################
# Request algorithm - Clustering
        
from algorithms.clustering.kmeans_clustering import kmeans_clustering
        
def request_clustering(data):
    X = data['features']
    clustering_alg = input('Select clustering algorithm:' +
        '\n\n 1. k-means clustering' +
        '\n\n Enter the option number: ')
    if (clustering_alg == '1'):
        return [kmeans_clustering(X)]
    elif (clustering_alg == '*'):
        models = []
        models.append(kmeans_clustering(X))
        return models
        
###############################################################################
# Request algorithm - Association Rule
        
from algorithms.association_rule.apriori import apriori

def request_association(data):   
    dataset = data['dataset']
    association_alg = input('Select clustering algorithm:' +
        '\n\n 1. apriori' 
        '\n\n Enter the option number: ')
    if (association_alg == '1'):
        return [apriori(dataset)]
    elif (association_alg == '*'):
        models = []
        models.append(apriori(dataset))
        return models
        