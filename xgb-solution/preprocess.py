import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import utils
from sklearn import preprocessing

def get_data():
    train = pd.read_csv('train_trendency.csv')
    test = pd.read_csv('test.csv')

    train['LogConfirmed'] = utils.to_log(train.Confirmed)
    train['LogDeaths'] = utils.to_log(train.Deaths)

    train['DateTime'] = pd.to_datetime(train['Date'])
    test['DateTime'] = pd.to_datetime(test['Date'])

    train = train.sort_values(by='Date')
    test = test.sort_values(by='Date')

    state_encoder = preprocessing.LabelEncoder()
    train['encoded_location'] = state_encoder.fit_transform(train.Province_State)

    return train, test


def process_by_state(data):
    dfs = []
    for loc, df in data.groupby('Province_State'):
        df = df.sort_values(by='Date')

        features = ['Deaths', 'Confirmed', 'LogDeaths', 'LogConfirmed']
        nextday_features = ['Deaths', 'Confirmed', 'LogDeaths', 'LogConfirmed', 'Date']

        for feature in features:
            df[f'{feature}'] = df[f'{feature}'].cummax()

        for feature in nextday_features: 
            df[f'{feature}NextDay'] = df[f'{feature}'].shift(-1)

        df['LogConfirmedDelta'] = df['LogConfirmedNextDay'] - df['LogConfirmed']
        df['LogDeathsDelta'] = df['LogDeathsNextDay'] - df['LogDeaths']
        dfs.append(df)
    return pd.concat(dfs)


def generate_rolling_features(data):
    dfs = []
    days_rolled = [1, 3, 7, 10, 21]
    for loc, df in data.groupby('Province_State'):
        df = df.sort_values(by='Date').copy()
        
        for t in days_rolled:
            df[f'logc_{t}d'] = df['LogConfirmed'].shift(t)
            df[f'logd_{t}d'] = df['LogDeaths'].shift(t)
        
        df['logc_0d'] = df['LogConfirmed']
        df['logd_0d'] = df['LogDeaths']
        df['dc_ratio'] = np.clip((utils.to_exp(df['LogDeaths']) + 1) / (utils.to_exp(df['LogConfirmed']) + 1), 0, 0.15)
        dfs.append(df)
    dfs = pd.concat(dfs)
    return dfs