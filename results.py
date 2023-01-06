from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, RocCurveDisplay
from statsmodels.stats.contingency_tables import mcnemar
from matplotlib import pyplot as plt
import numpy as np

from constants import SIGNIFICANCE_LEVEL

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

def results(y_test, y_pred_dtc, y_pred_knc):
    # plot the confusion matrix for the Decision Tree Classifier     
    classifier_cm(y_test, y_pred_dtc, 'Decision Tree Classifier')
    
    # plot the confusion matrix for the K-Neighbours Classifier
    classifier_cm(y_test, y_pred_knc, 'K-Neighbours Classifier')
    plt.show()

    contingency_table = createContingencyTable(y_test, y_pred_dtc, y_pred_knc)
    pvalue = signTest(contingency_table)

    significant = pvalue < SIGNIFICANCE_LEVEL

    print("P-value of McNemar sign test is {0}. At our significance level of {1}, this means that the performance difference between the classifiers is {2}.".format(pvalue, SIGNIFICANCE_LEVEL, "significant" if significant else "not significant"))

