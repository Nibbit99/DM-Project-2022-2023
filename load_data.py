from pandas import read_csv
import random

from constants import RANDOM_SEED, DATASET_SIZE, SAMPLE_SIZE

def getRowsToSkip(sample_size):
    # set random seed
    random.seed(RANDOM_SEED)

    skiprows = sorted(random.sample(range(1, DATASET_SIZE), DATASET_SIZE - sample_size))
    return skiprows

def load_data():
    # read random sample of the data
    dataset = read_csv('csgo_round_snapshots.csv', header=0, skiprows=getRowsToSkip(SAMPLE_SIZE))

    return dataset

    