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

# significance level
SIGNIFICANCE_LEVEL = 0.05