from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
import warnings
warnings.filterwarnings("ignore")

from constants import RANDOM_SEED, DTC_MIN_DEPTH, DTC_MAX_DEPTH, KNC_MIN_K, KNC_MAX_K

def splitData(X, X_standardized, y, train, test):
    return X[train], X[test], X_standardized[train], X_standardized[test], y[train], y[test]

def optimize_hyperparameters(classifier, hyperparameter, optimized_hyperparameter, optimized_accuracy, X_test, y_test):
    y_test_pred = classifier.predict(X_test)

    # get the accuracy for the classifier initialized with the given hyperparameter
    accuracy = accuracy_score(y_test, y_test_pred)

    # update optimized depth and its accuracy if its accuracy is greater than the previous best accuracy
    if accuracy > optimized_accuracy:
        optimized_accuracy = accuracy
        optimized_hyperparameter = hyperparameter

    # return the (possibly updated) values for the optimized accuracy and hyperparameter
    return optimized_hyperparameter, optimized_accuracy

def classification(X, X_standardized, y):
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

    return y_test, y_pred_dtc, y_pred_knc