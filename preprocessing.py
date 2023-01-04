from sklearn.preprocessing import OneHotEncoder, StandardScaler

def oneHotEncodeMap(data):
    # create One Hot Encoder
    ohe = OneHotEncoder(sparse=False) 
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

def encodeRoundWinner(team):
    return 1 if team == 'CT' else 0

def standardize(data):
    scaler = StandardScaler().fit(data)
    return scaler.transform(data)

def preprocessing(dataset):
    dataset['round_winner'] = dataset['round_winner'].apply(func=encodeRoundWinner)

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

    return X, X_standardized, y
