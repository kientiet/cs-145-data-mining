# Libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import datetime as dt
import xgboost as xgb
from sklearn import preprocessing

# Written code
import utils
import preprocess

def generate_prediction(TRAIN_START, TEST_START, TRAIN_END, TEST_END, DATEFORMAT = '%m-%d-%Y'):
    train, test = preprocess.get_data()
    train_clean = preprocess.process_by_state(train)
    train_clean = train_clean[[
        'Province_State', 'Date', 'LogConfirmed', 
        'LogDeaths', 'LogConfirmedDelta', 'LogDeathsDelta', 'encoded_location'
    ]]
    train_features = preprocess.generate_rolling_features(train_clean)

    features = [
        'logc_7d', 'logd_7d', 'logc_3d', 'logd_3d', 'encoded_location',
        'logc_1d', 'logd_1d', 'logc_0d', 'logd_0d', 'dc_ratio'
    ]

    config = dict(
        min_child_weight=5,
        eta=0.01, colsample_bytree=0.8, 
        max_depth=5, subsample=0.9, nthread=2, booster='gbtree',
        eval_metric='rmse', objective='reg:squarederror'
    )

    data = train_features[(train_features.Date >= TRAIN_START) & (train_features.Date < TRAIN_END)].copy()
    data['day_until'] = -(pd.to_datetime(train_features.Date) - dt.datetime.strptime(TRAIN_END, DATEFORMAT)).dt.days

    dm_logc = xgb.DMatrix(data[features].round(2), label=data.LogConfirmedDelta, weight=utils.calc_weight(data.day_until))
    dm_logd = xgb.DMatrix(data[features].round(2), label=data.LogDeathsDelta, weight=utils.calc_weight(data.day_until))

    model_lc = xgb.train(config, dm_logc, 800, evals=[(dm_logc, 'train-logc')], verbose_eval=100)
    model_lf = xgb.train(config, dm_logd, 800, evals=[(dm_logd, 'train-logd')], verbose_eval=100)

    # Predict
    predictions = data.copy()
    predictions = train_features[(train_features.Date >= TRAIN_START) & (train_features.Date <= TRAIN_END)].copy()
    predictions.LogConfirmedDelta = np.nan
    predictions.LogFatalitiesDelta = np.nan
    
    decay = 0.99
    for i, d in enumerate(pd.date_range(TRAIN_END, utils.add_days(TEST_END, 1)).strftime(DATEFORMAT)):
        last_day = str(d).split(' ')[0]
        next_day = dt.datetime.strptime(last_day, DATEFORMAT) + dt.timedelta(days=1)
        next_day = next_day.strftime(DATEFORMAT)

        p_next_day = predictions[predictions.Date == last_day].copy()
        p_next_day.Date = next_day
        p_next_day['p_logc'] = model_lc.predict(xgb.DMatrix(p_next_day[features].round(2)))
        p_next_day['p_logd'] = model_lf.predict(xgb.DMatrix(p_next_day[features].round(2)))

        p_next_day.LogConfirmed = p_next_day.LogConfirmed + np.clip(p_next_day['p_logc'], 0, None) * decay ** i
        p_next_day.LogDeaths = p_next_day.LogDeaths + np.clip(p_next_day['p_logd'], 0, None) * decay ** i

        predictions = pd.concat([predictions, p_next_day], sort=True)
        predictions = preprocess.generate_rolling_features(predictions)

    predictions['p_expc'] = utils.to_exp(predictions.LogConfirmed)
    predictions['p_expd'] = utils.to_exp(predictions.LogDeaths)
    return predictions