from load_data import load_data
from preprocessing import preprocessing
from classification import classification
from results import results

if __name__ == '__main__':
    dataset = load_data()

    X, X_standardized, y = preprocessing(dataset)

    y_test, y_pred_dtc, y_pred_knc = classification(X, X_standardized, y)

    results(y_test, y_pred_dtc, y_pred_knc)