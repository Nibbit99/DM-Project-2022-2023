from pandas import read_csv
import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from matplotlib import pyplot as plt
import random
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

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
    sample_size = 500 
    
    # set random seed
    random.seed(RANDOM_SEED)

    skiprows = sorted(random.sample(range(1, DATASET_SIZE), DATASET_SIZE - sample_size))
    return skiprows

def encodeRoundWinner(team):
    return 1 if team == 'CT' else 0

def splitData(X, X_standardized, y, train, test):
    return X[train], X[test], X_standardized[train], X_standardized[test], y[train], y[test]

def classifier_cm(y_test, y_pred, name):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['T', 'CT'])
    disp.plot()
    ac = accuracy_score(y_test,y_pred)
    plt.title('%s (AC: %s)' % (name, ac))

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

    # nested stratified 10-fold cross-validation
    total_splits = 2
    inner_cv = StratifiedKFold(n_splits=total_splits, shuffle=True, random_state=RANDOM_SEED)
    outer_cv = StratifiedKFold(n_splits=total_splits, shuffle=True, random_state=RANDOM_SEED)

    y_test = []
    y_pred_dtc = []
    y_pred_knc = []

    # outer cross validation: estimate classifier performance
    for train_outer, test_outer in outer_cv.split(X, y):
        X_train_outer, X_test_outer, X_standardized_train_outer, X_standardized_test_outer, y_train_outer, y_test_outer = splitData(X, X_standardized, y, train_outer, test_outer)
        y_test = [*y_test, *y_test_outer]

        # inner cross validation: optimize hyperparameters
        for train_inner, test_inner in inner_cv.split(X_test_outer, y_test_outer):
            X_train_inner, X_test_inner, X_standardized_train_inner, X_standardized_test_inner, y_train_inner, y_test_inner = splitData(X_test_outer, X_standardized_test_outer, y_test_outer, train_inner, test_inner)

            # TODO optimize for dtc: criterion, max_depth, max_depth?

            # TODO optimize for knn: n_neighbors, metric

        # Decision Tree Classifier
        dtc = DecisionTreeClassifier(random_state=RANDOM_SEED)
        dtc.fit(X_train_outer, y_train_outer)
        
        y_test_pred_dtc = dtc.predict(X_test_outer)
        y_pred_dtc = [*y_pred_dtc, *y_test_pred_dtc]
        
        # K-Neighbours Classifier
        knc = KNeighborsClassifier(n_neighbors=2)
        knc.fit(X_standardized_train_outer, y_train_outer)
        
        y_test_pred_knc = knc.predict(X_standardized_test_outer)
        y_pred_knc = [*y_pred_knc, *y_test_pred_knc]
       
    # plot the confusion matrix for the Decision Tree Classifier     
    classifier_cm(y_test, y_pred_dtc, 'Decision Tree Classifier')
    
    # plot the confusion matrix for the K-Neighbours Classifier
    classifier_cm(y_test, y_pred_knc, 'K-Neighbours Classifier')
    plt.show()