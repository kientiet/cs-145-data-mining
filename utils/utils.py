import pandas as pd
import numpy as np

from typing import Any, Callable, List, Tuple


def split_features_label(dataset: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
    confirmed = dataset["Confirmed"]
    deaths = dataset["Deaths"]
    dataset = dataset.drop(columns=["Confirmed", "Deaths"])

    return dataset, confirmed, deaths


def train_test_split(
    dataset: pd.DataFrame, drop_columns: List[str]
) -> Tuple[np.ndarray, ...]:
    new_dataset = dataset.copy(deep=True)
    new_dataset["month"] = new_dataset["Date"].map(lambda date: int(date[:2]))

    val_dataset = new_dataset[new_dataset["month"] == 3]
    train_dataset = new_dataset[new_dataset["month"] != 3]

    train_dataset = train_dataset.drop(columns=drop_columns + ["month"])
    val_dataset = val_dataset.drop(columns=drop_columns + ["month"])

    val_features, val_confirmed, val_deaths = split_features_label(val_dataset)
    train_features, train_confirmed, train_deaths = split_features_label(train_dataset)

    return (
        train_features.values,
        val_features.values,
        train_confirmed.values,
        val_confirmed.values,
        train_deaths.values,
        val_deaths.values,
    )


def torch_train_test_split(
    dataset: pd.DataFrame, drop_columns: List[str]
) -> Tuple[np.ndarray, ...]:
    new_dataset = dataset.copy(deep=True)
    new_dataset["month"] = new_dataset["Date"].map(lambda date: int(date[:2]))

    val_dataset = new_dataset[new_dataset["month"] == 3]
    train_dataset = new_dataset[new_dataset["month"] != 3]

    train_dataset = train_dataset.drop(columns=drop_columns + ["month"])
    val_dataset = val_dataset.drop(columns=drop_columns + ["month"])

    return train_dataset, val_dataset


def cumulative(data: np.ndarray) -> np.ndarray:
    return np.cumsum(data, axis=1)


def heuristic_smooth(data: np.array) -> np.array:
    new_data = np.roll(data, 1)
    data[0] = 0

    result = 0.66 * data + 0.33 * new_data

    return result


def mean_absolute_percentage_error(
    labels: np.ndarray, predictions: np.ndarray
) -> float:
    total = (
        np.sum(np.abs(labels - predictions) / labels, axis=0) / labels.shape[0] * 100.0
    )

    return total.sum()

def evaluation_prediction(
    confirmed_prediction: np.ndarray,
    deaths_prediction: np.ndarray,
    confirmed: np.ndarray,
    deaths: np.ndarray,
    smooth_function: Callable[..., Any] = None,
) -> float:
    confirmed_prediction = np.array(confirmed_prediction).transpose()
    deaths_prediction = np.array(deaths_prediction).transpose()

    if smooth_function is not None:
        confirmed_prediction = smooth_function(confirmed_prediction)
        deaths_prediction = smooth_function(deaths_prediction)

    confirmed = cumulative(confirmed)
    deaths = cumulative(deaths)

    confirmed_prediction = confirmed_prediction.flatten("F")
    deaths_prediction = deaths_prediction.flatten("F")
    confirmed = confirmed.flatten("F")
    deaths = deaths.flatten("F")

    labels = np.column_stack((confirmed, deaths))
    predictions = np.column_stack((confirmed_prediction, deaths_prediction))

    return mean_absolute_percentage_error(labels, predictions)
