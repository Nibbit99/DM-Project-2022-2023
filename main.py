from pandas import read_csv
import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from matplotlib import pyplot as plt
import random
from sklearn.model_selection import StratifiedKFold

import warnings
warnings.filterwarnings("ignore")

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

def encodeRoundWinner(team):
    return 1 if team == 'CT' else 0

if __name__ == '__main__':
    # read random sample of the data
    dataset = read_csv('csgo_round_snapshots.csv', header=0, skiprows=getRowsToSkip(100))
    dataset['round_winner'] = dataset['round_winner'].apply(func=encodeRoundWinner)
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
    ten_fold = StratifiedKFold(n_splits=2, shuffle=True, random_state=RANDOM_SEED)
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
        knc = KNeighborsClassifier(n_neighbors=2)
        knc.fit(X_standardized_train, y_train)
        
        y_test_pred = knc.predict(X_standardized_test)
        
        correct = len(y_test[y_test == y_test_pred])
        accuracy = correct/len(y_test)
        
        