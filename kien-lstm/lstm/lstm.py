import os
import torch
import math
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from tqdm import tqdm
from typing import Tuple
from lstm.dataset import CustomDataset
from torch.utils.data import Dataset, DataLoader
from utils.utils import (
    cumulative,
    split_features_label,
    torch_train_test_split,
    evaluation_prediction,
)

num_epochs = 100
hidden_size = 128
num_layers = 2

device = "cuda" if torch.cuda.is_available() else "cpu"


class CustomLSTM(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.2,
            batch_first=True,
        )
        self.hidden_size = hidden_size
        self.projection = nn.Linear(hidden_size, num_features)
        self.activation = nn.ReLU()

        self.__init_weight()

    def __init_weight(self):
        std = math.sqrt(4.0 / self.projection.weight.size(0))
        nn.init.normal_(self.projection.weight, mean=0, std=std)
        nn.init.constant_(self.projection.bias, 0)

    def forward(self, inputs: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        inputs = inputs.unsqueeze(1)
        outputs, (hn, cn) = self.encoder(inputs, hidden)
        outputs = self.projection(self.activation(outputs))
        return outputs.squeeze(), (hn, cn)

    def init_hidden(self) -> Tuple[torch.Tensor]:
        return torch.zeros(1, 1, self.hidden_size, device=device)


class LSTMTrainer:
    def __init__(
        self, train_dataset: pd.DataFrame, val_dataset: pd.DataFrame, filename: str
    ) -> None:
        if val_dataset is None:
            train_dataset, val_dataset = torch_train_test_split(
                train_dataset,
                drop_columns=["Recovered", "Active"],
            )
        else:
            columns = [
                "Recovered",
                "Active",
                "Incident_Rate",
                "Total_Test_Results",
                "Case_Fatality_Ratio",
                "Testing_Rate",
            ]
            train_dataset, val_dataset = train_dataset.drop(
                columns=columns
            ), val_dataset.drop(columns=columns)

            order_columns = [
                "Province_State",
                "Date",
                "population",
                "longitude",
                "latitude",
                "Confirmed",
                "Deaths",
            ]
            train_dataset = train_dataset[order_columns]
            val_dataset = val_dataset[order_columns]

        self.filename = filename
        self.train_dataset = CustomDataset(train_dataset)

        val_dataset = val_dataset.reset_index(drop=True)
        self.val_dataset = CustomDataset(val_dataset)

    def __init_model(self) -> None:
        num_features = self.train_dataset.batch_dataset[0].shape[1]
        self.encoder = CustomLSTM(num_features=num_features)

    def __init_dataloader(self):
        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=50, shuffle=False, num_workers=8
        )

        self.val_dataloader = DataLoader(
            self.val_dataset, batch_size=50, shuffle=False, num_workers=8
        )

    def __train(self, save_prediction: bool = False):
        self.encoder = self.encoder.to(device)

        optimizer = optim.Adam(self.encoder.parameters(), lr=0.05, weight_decay=1e-5)

        loss_func = nn.L1Loss()
        total_loss = []
        performance = []

        for epoch in tqdm(range(num_epochs)):
            total_loss.append([0, 0])

            # ? Init the requirements
            """
                TODO: Try initialize as same distribution
            """
            encoder_input = torch.randn(
                (50, self.train_dataset.batch_dataset[0].shape[1]),
                requires_grad=True,
                device=device,
            ).float()

            encoder_hidden = None

            for index, dataloader in enumerate(
                [self.train_dataloader, self.val_dataloader]
            ):
                self.encoder = (
                    self.encoder.eval() if index == 1 else self.encoder.train()
                )

                for features in dataloader:
                    features = features.float().to(device)

                    confirmed_prediction, deaths_prediction = [], []

                    loss = 0
                    for time_step in range(features.shape[1]):
                        encoder_output, encoder_hidden = self.encoder(
                            encoder_input, encoder_hidden
                        )

                        loss += loss_func(encoder_output, features[:, time_step, :])
                        encoder_input = encoder_output.detach()

                        with torch.no_grad():
                            confirmed_prediction.append(
                                # self.train_dataset.inverse_confirmed(
                                encoder_input[:, -2]
                                .cpu()
                                .numpy()
                                # )
                            )
                            deaths_prediction.append(
                                # self.train_dataset.inverse_deaths(
                                encoder_input[:, -1]
                                .cpu()
                                .numpy()
                                # )
                            )

                    if index:
                        with torch.no_grad():
                            total_loss[epoch][1] = loss.item()
                            # confirmed = self.train_dataset.inverse_confirmed(
                            #     features[:, :, -2].cpu().numpy()
                            # )
                            # deaths = self.train_dataset.inverse_deaths(
                            #     features[:, :, -1].cpu().numpy()
                            # )

                            confirmed = features[:, :, -2].cpu().numpy()

                            deaths = features[:, :, -1].cpu().numpy()

                            performance.append(
                                evaluation_prediction(
                                    confirmed_prediction,
                                    deaths_prediction,
                                    confirmed,
                                    deaths,
                                    smooth_function=cumulative,
                                )
                            )
                    else:
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        total_loss[epoch][0] = loss.item()

        torch.save(self.encoder.state_dict(), self.filename)
        # ? Plot performance
        total_loss, performance = np.array(total_loss), np.array(performance)
        fig, axs = plt.subplots(2)

        axs[0].plot(total_loss[5:, 0], label="train_loss")
        axs[0].plot(total_loss[5:, 1], label="val_loss")
        axs[1].plot(performance, label="MAPE")
        axs[0].legend()
        axs[1].legend()
        plt.show()
        print("Final performance")
        print(total_loss[-1])
        print(performance[-1])

        if save_prediction:
            self.__print_prediction(confirmed_prediction, deaths_prediction)

    def __print_prediction(
        self, confirmed_prediction: np.ndarray, deaths_prediction: np.ndarray
    ) -> None:
        confirmed_prediction = np.array(confirmed_prediction)
        deaths_prediction = np.array(deaths_prediction)

        test_dataframe = pd.read_csv(
            os.path.join(os.getcwd(), "ucla-covid19-prediction", "test.csv"),
            index_col=0,
        )
        current_date, current_index = test_dataframe.iloc[0]["Date"], 0
        for index, row in test_dataframe.iterrows():
            predict_province = row["Province_State"]
            batch_id = np.where(self.train_dataset.province == predict_province)[0][0]

            if current_date != row["Date"]:
                current_date = row["Date"]
                current_index += 1

            test_dataframe.at[index, "Confirmed"] = np.sum(
                confirmed_prediction[: (current_index + 1), batch_id]
            )

            test_dataframe.at[index, "Deaths"] = np.sum(
                deaths_prediction[: (current_index + 1), batch_id]
            )

        submission = pd.read_csv(
            os.path.join(os.getcwd(), "ucla-covid19-prediction", "submission.csv"),
            index_col=0,
        )
        submission["Confirmed"] = test_dataframe["Confirmed"]
        submission["Deaths"] = test_dataframe["Deaths"]
        submission.to_csv("team2.csv")

    def fit(self, save_prediction: bool = False) -> None:
        self.__init_model()
        self.__init_dataloader()
        self.__train(save_prediction)
