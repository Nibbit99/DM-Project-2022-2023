from pandas import read_csv
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, plot_tree
from matplotlib import pyplot as plt
from toolbox.clusterPlot import clusterPlot
import random

SAMPLE_SIZE = 100
DATASET_SIZE = 122410

def oneHotEncodeMap(data):
    # create One Hot Encoder
    ohe = preprocessing.OneHotEncoder(sparse=False) 
    # One Hot Encode map attribute
    map_OneHotEncoded = ohe.fit_transform(data[['map']])

    # list of all map categories
    maps = ohe.categories_[0]

    # remove original map attribute
    data.drop(columns='map', inplace=True)

    # iterate over all maps
    for i in range(maps.size):
        # get map name
        map = maps[i]
        # insert attributes after the fourth column
        data.insert(3+i, map, map_OneHotEncoded[:, i])

    return data

def standardize(data):
    scaler = preprocessing.StandardScaler().fit(X)
    return scaler.transform(X)

def getRowsToSkip(sample_size):
    sample_size = 100 #desired sample size
    skiprows = sorted(random.sample(range(1, DATASET_SIZE), DATASET_SIZE - sample_size))
    return skiprows

if __name__ == '__main__':
    # read random sample of the data
    dataset = read_csv('csgo_round_snapshots.csv', nrows=110, header=0, skiprows=getRowsToSkip(100))
    print(dataset)

    # attributes without class label
    X = dataset.drop(columns='round_winner')
    # class label
    y = dataset['round_winner']

    # One Hot Encode the map attribute
    X = oneHotEncodeMap(X)

    # transform X and y to nparrays
    X = X.to_numpy()
    y = y.to_numpy()

    # standardize data
    X_standardized = standardize(X)

    # Decision Tree Classifier
    dtc = DecisionTreeClassifier(random_state=0)
    dtc.fit(X, y)

    plt.figure(figsize=(20,20))
    plot_tree(dtc, node_ids=True)
    plt.show()

    # K-Means Clustering
    kmeans = KMeans(2, random_state=0)
    kmeans.fit(X_standardized)
    y_pred = kmeans.predict(X_standardized)
    clusters = kmeans.labels_
    centroids = kmeans.cluster_centers_
    
    clusterPlot(X_standardized, clusters, centroids, y)
    plt.show()
    