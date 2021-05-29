import torch
import numpy as np
import pandas as pd

from typing import Tuple
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, dataset: pd.DataFrame) -> None:
        super().__init__()
        self.dataset = dataset
        self.__dataframe_to_time_series(self.dataset.copy(deep=True))

    def __preprocessing(self, dataset: pd.DataFrame) -> pd.DataFrame:
        categorical_features = ["Province_State", "Date"]
        new_dataset = dataset.drop(columns=categorical_features)
        columns = new_dataset.columns

        self.scaler = MinMaxScaler().fit(new_dataset.values)
        transformed_data = self.scaler.transform(new_dataset.values)
        new_dataset = pd.DataFrame(data=transformed_data, columns=columns)

        dataset.loc[:, columns] = new_dataset.loc[:, columns]
        return dataset

    def __dataframe_to_time_series(self, dataset: pd.DataFrame) -> np.ndarray:
        dataset = dataset.sort_values(by=["Province_State", "Date"])

        self.new_dataset = self.__preprocessing(dataset.copy(deep=True))
        self.province = self.new_dataset["Province_State"].unique()

        state_group = self.new_dataset.groupby(by="Province_State")

        self.batch_dataset = []
        for index, group in state_group:
            features = group.drop(columns=["Province_State", "Date"])
            self.batch_dataset.append(features.values)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, ...]:
        return self.batch_dataset[index].astype(float)

    def __len__(self):
        return len(self.batch_dataset)
