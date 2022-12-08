from pandas import read_csv
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, plot_tree
from matplotlib import pyplot as plt
from toolbox.clusterPlot import clusterPlot
import random
from sklearn.model_selection import StratifiedKFold

# size of sample we want to read from dataset
SAMPLE_SIZE = 100
# size of dataset
DATASET_SIZE = 122410
# random seed used to reproduce results
RANDOM_SEED = 0

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
    # desired sample size
    sample_size = 100 
    
    # set random seed
    random.seed(RANDOM_SEED)

    skiprows = sorted(random.sample(range(1, DATASET_SIZE), DATASET_SIZE - sample_size))
    return skiprows

if __name__ == '__main__':
    # read random sample of the data
    dataset = read_csv('csgo_round_snapshots.csv', header=0, skiprows=getRowsToSkip(100))
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

    # Stratified 10-fold
    ten_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
    for train, test in ten_fold.split(X, y):
        X_train, X_test, X_standardized_train, X_standardized_test, y_train, y_test = X[train], X[test], X_standardized[train], X_standardized[test], y[train], y[test]

        # Decision Tree Classifier
        dtc = DecisionTreeClassifier(random_state=RANDOM_SEED)
        dtc.fit(X_train, y_train)

        """
        plt.figure(figsize=(20,20))
        plot_tree(dtc, node_ids=True)
        plt.show()
        """
        
        # K-Means Clustering
        kmeans = KMeans(2, random_state=RANDOM_SEED)
        kmeans.fit(X_standardized_train)
        # y_pred = kmeans.predict(X_standardized)
        clusters = kmeans.labels_
        centroids = kmeans.cluster_centers_
        
        """
        clusterPlot(X_standardized_test, clusters, centroids, y)
        plt.show()
        """
    
    