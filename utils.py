import pandas as pd
import numpy as np

from typing import Any, Callable, List, Tuple
from sklearn.metrics import mean_absolute_percentage_error


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


def cumulative(data: np.ndarray) -> np.ndarray:
    return np.cumsum(data)


def evaluation_prediction(
    predictions: np.ndarray,
    labels: np.ndarray,
    smooth_function: Callable[..., Any] = None,
) -> float:
    if smooth_function is not None:
        predictions = smooth_function(predictions)

    labels = cumulative(labels)

    return mean_absolute_percentage_error(labels, predictions), predictions, labels
