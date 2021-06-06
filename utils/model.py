import pandas as pd
import numpy as np

from typing import Any, Callable, List, Tuple, Union
from utils.utils import (
    split_features_label,
    train_test_split,
    cumulative,
    evaluation_prediction,
)


def learning_process_with_full_dataset(
    dataset: pd.DataFrame, model: Any, smooth_function: Callable[..., Any]
) -> Tuple[Union[List[float], np.ndarray]]:
    states = dataset["Province_State"].unique()
    total_score = [0, 0]

    for state in states:
        state_dataset = dataset[dataset["Province_State"] == state].copy(deep=True)

        state_dataset.drop(
            columns=["Date", "Confirmed", "Deaths", "Recovered", "Active"]
        )

        (
            train_features,
            val_features,
            train_confirmed,
            val_confirmed,
            train_deaths,
            val_deaths,
        ) = train_test_split(
            state_dataset,
            drop_columns=["Recovered", "Active", "Province_State", "Date"],
        )

        target = [[train_confirmed, val_confirmed], [train_deaths, val_deaths]]
        for index, label_pair in enumerate(target):
            clf = model.fit(train_features, label_pair[0])
            preds = clf.predict(val_features)
            score, new_predictions, new_labels = evaluation_prediction(
                preds, label_pair[1], smooth_function=smooth_function
            )
            total_score[index] += score

    return total_score, new_predictions, new_labels


def get_data_with_state(
    dataset: pd.DataFrame, state: str, drop_columns: List[str]
) -> pd.DataFrame:
    state_dataset = dataset[dataset["Province_State"] == state].copy(deep=True)
    state_dataset = state_dataset.drop(columns=drop_columns)

    features, confirmed, deaths = split_features_label(state_dataset)

    return features, confirmed, deaths


def learning_process_with_external_test(
    train_dataset: pd.DataFrame,
    test_dataset: pd.DataFrame,
    model: Any,
    smooth_function: Callable[..., Any],
    drop_columns: List[str] = ["Recovered", "Active", "Province_State", "Date"],
) -> Tuple[Union[List[float], np.ndarray]]:
    states = train_dataset["Province_State"].unique()
    total_score = [0, 0]

    for state in states:
        train_features, train_confirmed, train_deaths = get_data_with_state(
            train_dataset, state=state, drop_columns=drop_columns
        )

        test_features, test_confirmed, test_deaths = get_data_with_state(
            test_dataset, state=state, drop_columns=drop_columns
        )

        target = [[train_confirmed, test_confirmed], [train_deaths, test_deaths]]
        for index, label_pair in enumerate(target):
            clf = model.fit(train_features, label_pair[0])
            preds = clf.predict(test_features)
            score, new_predictions, new_labels = evaluation_prediction(
                preds, label_pair[1], smooth_function=smooth_function
            )
            total_score[index] += score

    return total_score, new_predictions, new_labels
