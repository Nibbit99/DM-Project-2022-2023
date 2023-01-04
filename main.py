from pandas import read_csv
import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from matplotlib import pyplot as plt
import random
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

from statsmodels.stats.contingency_tables import mcnemar

import warnings
warnings.filterwarnings("ignore")

"""CONSTANTS"""
# size of sample we want to read from dataset
SAMPLE_SIZE = 100
# size of dataset
DATASET_SIZE = 122410

# random seed used to reproduce results
RANDOM_SEED = 0

# lower bound on depth values to try for decision tree classifier
DTC_MIN_DEPTH = 10
# upper bound on depth values to try for decision tree classifier
DTC_MAX_DEPTH = 30

# lower bound on k values to try for the k neighbors classifier
KNC_MIN_K = 1
# upper bound on k values to try for the k neighbors classifier
KNC_MAX_K = 20

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

def createContingencyTable(true_class, pred_clf1, pred_clf2):
    table = np.zeros([2,2])

    # index (0, 0) stores how many times both classifiers predicted the correct label
    table[0][0] = sum(t == p1 == p2 for t, p1, p2 in zip(true_class, pred_clf1, pred_clf2))
    # index (1, 1) stores how many times both classifiers predicted the wrong label
    table[1][1] = sum(t == (1-p1) == (1-p2) for t, p1, p2 in zip(true_class, pred_clf1, pred_clf2))
    # index (1, 0) stores how many times classifier 2 predicted the correct label, but classifier 1 predicted the wrong label
    table[1][0] = sum(t == (1-p1) == p2 for t, p1, p2 in zip(true_class, pred_clf1, pred_clf2))
    # index (0, 1) stores how many times classifier 1 predicted the correct label, but  classifier 2 predicted the wrong label
    table[0][1] = sum(t == p1 == (1-p2) for t, p1, p2 in zip(true_class, pred_clf1, pred_clf2))

    return table


def signTest(contingency_table):
    return mcnemar(contingency_table, exact=False, correction=False).pvalue

def optimize_hyperparameters(classifier, hyperparameter, optimized_hyperparameter, optimized_accuracy, X_test, y_test, ):
    y_test_pred = classifier.predict(X_test)

    # get the accuracy for the classifier initialized with the given hyperparameter
    accuracy = accuracy_score(y_test, y_test_pred)

    # update optimized depth and its accuracy if its accuracy is greater than the previous best accuracy
    if accuracy > optimized_accuracy:
        optimized_accuracy = accuracy
        optimized_hyperparameter = hyperparameter

    # return the (possibly updated) values for the optimized accuracy and hyperparameter
    return optimized_hyperparameter, optimized_accuracy

if __name__ == '__main__':
    # read random sample of the data
    dataset = read_csv('csgo_round_snapshots.csv', header=0, skiprows=getRowsToSkip(SAMPLE_SIZE))
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
    total_splits = 10
    inner_cv = StratifiedKFold(n_splits=total_splits, shuffle=True, random_state=RANDOM_SEED)
    outer_cv = StratifiedKFold(n_splits=total_splits, shuffle=True, random_state=RANDOM_SEED)

    y_test = []
    y_pred_dtc = []
    y_pred_knc = []

    # outer cross validation: estimate classifier performance
    for train_outer, test_outer in outer_cv.split(X, y):
        X_train_outer, X_test_outer, X_standardized_train_outer, X_standardized_test_outer, y_train_outer, y_test_outer = splitData(X, X_standardized, y, train_outer, test_outer)
        y_test = [*y_test, *y_test_outer]

        # TODO optimize for dtc: criterion, max_depth, max_depth?
        dtc_optimized_depth = DTC_MIN_DEPTH
        dtc_optimized_accuracy = 0

        # TODO optimize for knn: n_neighbors, metric
        knc_optimized_k = KNC_MIN_K
        knc_optimized_accuracy = 0

        # inner cross validation: optimize hyperparameters
        for train_inner, test_inner in inner_cv.split(X_train_outer, y_train_outer):
            X_train_inner, X_test_inner, X_standardized_train_inner, X_standardized_test_inner, y_train_inner, y_test_inner = splitData(X_train_outer, X_standardized_train_outer, y_train_outer, train_inner, test_inner)

            # optimize depth for decision tree classifier
            for d in range(DTC_MIN_DEPTH, DTC_MAX_DEPTH + 1):
                dtc = DecisionTreeClassifier(random_state=RANDOM_SEED, max_depth = d)
                dtc.fit(X_train_inner, y_train_inner)

                dtc_optimized_depth, dtc_optimized_accuracy = optimize_hyperparameters(dtc, d, dtc_optimized_depth, dtc_optimized_accuracy, X_test_inner, y_test_inner)

            # optimize k for k neighbors classifiers
            for k in range(KNC_MIN_K, KNC_MAX_K + 1):
                knc = KNeighborsClassifier(n_neighbors=k)
                knc.fit(X_standardized_train_inner, y_train_inner)
                
                knc_optimized_k, knc_optimized_accuracy = optimize_hyperparameters(knc, k, knc_optimized_k, knc_optimized_accuracy, X_standardized_test_inner, y_test_inner)
        
        # Decision Tree Classifier
        dtc = DecisionTreeClassifier(random_state=RANDOM_SEED, max_depth = dtc_optimized_depth)
        dtc.fit(X_train_outer, y_train_outer)
        
        y_test_pred_dtc = dtc.predict(X_test_outer)
        y_pred_dtc = [*y_pred_dtc, *y_test_pred_dtc]
        
        # K-Neighbours Classifier
        knc = KNeighborsClassifier(n_neighbors=knc_optimized_k)
        knc.fit(X_standardized_train_outer, y_train_outer)
        
        y_test_pred_knc = knc.predict(X_standardized_test_outer)
        y_pred_knc = [*y_pred_knc, *y_test_pred_knc]
       
    # plot the confusion matrix for the Decision Tree Classifier     
    classifier_cm(y_test, y_pred_dtc, 'Decision Tree Classifier')
    
    # plot the confusion matrix for the K-Neighbours Classifier
    classifier_cm(y_test, y_pred_knc, 'K-Neighbours Classifier')
    plt.show()

    contingency_table = createContingencyTable(y_test, y_pred_dtc, y_pred_knc)
    pvalue = signTest(contingency_table)
    print(pvalue)