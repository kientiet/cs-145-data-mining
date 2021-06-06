import os
import pandas as pd
import numpy as np

from typing import Any, Callable, List, Tuple, Union
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.forecasting.stl import STLForecast

# need to add util functions here because sklearn won't update lol


def split_features_label(dataset: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
    cases = dataset["Cases"]
    deaths = dataset["Deaths"]
    dataset = dataset.drop(columns=["Cases", "Deaths"])
    return dataset, cases, deaths


def train_test_split(
    dataset: pd.DataFrame, drop_columns: List[str]
) -> Tuple[np.ndarray, ...]:
    new_dataset = dataset.copy(deep=True)
    new_dataset["month"] = new_dataset["Date"].map(lambda date: int(date[:2]))

    val_dataset = new_dataset[new_dataset["month"] == 3]
    train_dataset = new_dataset[new_dataset["month"] != 3]

    train_dataset = train_dataset.drop(columns=drop_columns + ["month"])
    val_dataset = val_dataset.drop(columns=drop_columns + ["month"])

    val_features, val_cases, val_deaths = split_features_label(val_dataset)
    train_features, train_cases, train_deaths = split_features_label(train_dataset)

    return (
        train_features.values,
        val_features.values,
        train_cases.values,
        val_cases.values,
        train_deaths.values,
        val_deaths.values,
    )


def mean_absolute_percentage_error(labels, predictions):
    sum_ape = 0
    N = 0
    for label, prediction in zip(labels, predictions):
        ape = np.abs(label - prediction) / label
        sum_ape += ape
        N += 1
    mape = sum_ape / N
    return mape


def get_data_with_state(
    dataset: pd.DataFrame, state: str) -> pd.DataFrame:
    state_dataset = dataset[dataset["Province_State"] == state]
    state_series = state_dataset.set_index('Date')
    cases = state_series['Confirmed']
    deaths = state_series['Deaths']
    return cases, deaths


def arima_test(
    train_dataset: pd.DataFrame,
    test_dataset: pd.DataFrame,
    case_order: Tuple,
    death_order: Tuple):

    data_dir = os.path.join(os.getcwd(), "../ucla-covid19-prediction")
    id_table = pd.read_csv(os.path.join(data_dir, "test.csv"), index_col=0)
    id_table['Date'] = pd.to_datetime(id_table['Date'])
    output = []

    states = train_dataset["Province_State"].unique()
    total_score = [0, 0]

    for state in states:
        train_cases, train_deaths = get_data_with_state(train_dataset, state=state)
        test_cases, test_deaths = get_data_with_state(test_dataset, state=state)

        cases_model = STLForecast(train_cases, ARIMA, period=7, model_kwargs={"order": case_order})
        cases_fit = cases_model.fit()
        pred_cases = cases_fit.forecast(30)
        pred_cases.index = pd.date_range(start='2021-04-01', end='2021-04-30')
        cases_score = mean_absolute_percentage_error(test_cases, pred_cases)

        deaths_model = STLForecast(train_deaths, ARIMA, period=7, model_kwargs={"order": death_order})
        deaths_fit = deaths_model.fit()
        pred_deaths = deaths_fit.forecast(30)
        pred_deaths.index = pd.date_range(start='2021-04-01', end='2021-04-30')
        deaths_score = mean_absolute_percentage_error(test_deaths, pred_deaths)

        for day_cases, day_deaths in zip(pred_cases, pred_deaths):
            day = pred_cases[pred_cases==day_cases].index[0]
            id = id_table[(id_table['Province_State'] == state) & (id_table['Date'] == day)].index[0]
            output.append([id, day_cases, day_deaths])
        total_score[0] += cases_score
        total_score[1] += deaths_score
    output = pd.DataFrame(output, columns=['ID', 'Confirmed', 'Deaths'])
    output.sort_values('ID', inplace=True)
    return output, total_score
