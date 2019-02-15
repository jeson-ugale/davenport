# K-Means Clustering

from time import time
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np    

def kmeans_clustering(features):
    start=time()
    
    # Using the elbow method to find the optimal number of clusters
    X = features
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    
    # Calculating optimal clusters
    p1 = np.array([1, wcss[0]])
    p2 = np.array([len(wcss), wcss[-1]])
    distances = []
    for i in range(0, len(wcss)):
        p3 = np.array([i, wcss[i]])
        dist = abs(np.cross(p2 - p1, p3 - p1)/np.linalg.norm(p2 - p1))
        distances.append(dist)
    opt_clusters = np.argmax(distances) + 1
    
    # Fitting K-Means to the dataset
    kmeans = KMeans(n_clusters = opt_clusters, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    y_pred = kmeans.predict(X)
    y_pred = pd.Series(y_pred)
    y_pred = y_pred.rename('Predicted Cluster')
    return {'name':'K-Means Clustering','model':kmeans, 'opt_clusters':opt_clusters, 'predictions':y_pred, 'duration':(time()-start)/60}