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

def splitData(X, X_standardized, y, train, test):
    return X[train], X[test], X_standardized[train], X_standardized[test], y[train], y[test]

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

    # nested stratified 10-fold cross-validation
    inner_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
    outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)

    # outer cross validation: estimate classifier performance
    for train_outer, test_outer in outer_cv.split(X, y):
        X_train_outer, X_test_outer, X_standardized_train_outer, X_standardized_test_outer, y_train_outer, y_test_outer = splitData(X, X_standardized, y, train_outer, test_outer)

        # inner cross validation: optimize hyperparameters
        for train_inner, test_inner in inner_cv.split(X_test_outer, y_test_outer):
            X_train_inner, X_test_inner, X_standardized_train_inner, X_standardized_test_inner, y_train_inner, y_test_inner = splitData(X_test_outer, X_standardized_test_outer, y_test_outer, train_inner, test_inner)

            # TODO optimize for dtc: criterion, max_depth, max_depth?

            # TODO optimize for knn: n_neighbors, metric

        """
        # Decision Tree Classifier
        dtc = DecisionTreeClassifier(random_state=RANDOM_SEED)
        dtc.fit(X_train, y_train)


        
        # K-Means Clustering
        kmeans = KMeans(2, random_state=RANDOM_SEED)
        kmeans.fit(X_standardized_train)
        y_pred = kmeans.predict(X_standardized_test)
        clusters = kmeans.labels_
        centroids = kmeans.cluster_centers_

        """
    