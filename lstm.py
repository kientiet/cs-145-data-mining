from typing import Tuple
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from dataset import CustomDataset
from torch.utils.data import Dataset, DataLoader
from utils import (
    cumulative,
    split_features_label,
    torch_train_test_split,
    evaluation_prediction,
)

num_epochs = 30
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

    def forward(self, inputs: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        inputs = inputs.unsqueeze(1)
        outputs, (hn, cn) = self.encoder(inputs, hidden)
        outputs = self.projection(self.activation(outputs))
        return outputs.squeeze(), (hn, cn)

    def init_hidden(self) -> Tuple[torch.Tensor]:
        return torch.zeros(1, 1, self.hidden_size, device=device)


class LSTMTrainer:
    def __init__(self, train_dataset: pd.DataFrame, val_dataset: pd.DataFrame) -> None:
        if val_dataset is None:
            train_dataset, val_dataset = torch_train_test_split(
                train_dataset,
                drop_columns=["Recovered", "Active"],
            )

        self.train_dataset = CustomDataset(train_dataset)
        self.val_dataset = CustomDataset(val_dataset)

    def __init_model(self) -> None:
        num_features = self.train_dataset.batch_dataset[0].shape[1]
        self.encoder = CustomLSTM(num_features=num_features)

    def __init_dataloader(self):
        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=50, shuffle=False, num_workers=8
        )

        self.val_dataloader = DataLoader(
            self.train_dataset, batch_size=50, shuffle=False, num_workers=8
        )

    def __train(self):
        self.encoder = self.encoder.to(device)

        optimizer = optim.Adam(self.encoder.parameters(), lr=0.05, weight_decay=1e-4)

        loss_func = nn.MSELoss()
        total_loss = []
        performance = []

        for epoch in range(num_epochs):
            print(f">> At epoch {epoch}")
            total_loss.append([0, 0])
            performance.append([0, 0])

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
                                encoder_input[:, -2].cpu().numpy()
                            )
                            deaths_prediction.append(encoder_input[:, -1].cpu().numpy())

                    if index:
                        with torch.no_grad():
                            total_loss[epoch][1] = loss.item()
                            confirmed, deaths = features[:, :, -2], features[:, :, -1]
                            performance[epoch][0] = evaluation_prediction(
                                np.array(confirmed_prediction),
                                confirmed.cpu().numpy(),
                                smooth_function=cumulative,
                            )[0]

                            performance[epoch][1] = evaluation_prediction(
                                np.array(deaths_prediction),
                                deaths.cpu().numpy(),
                                smooth_function=cumulative,
                            )[0]
                    else:
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        total_loss[epoch][0] = loss.item()

        # ? Plot performance
        total_loss, performance = np.array(total_loss), np.array(performance)
        fig, axs = plt.subplots(2)

        axs[0].plot(total_loss[1:, 0], label="train_loss")
        axs[0].plot(total_loss[1:, 1], label="val_loss")
        axs[1].plot(performance[:, 0], label="MAPE for confirmed")
        axs[1].plot(performance[:, 1], label="MAPE for deaths")
        axs[0].legend()
        axs[1].legend()
        plt.show()
        print("Final performance")
        print(performance[-1])

    def fit(self) -> None:
        self.__init_model()
        self.__init_dataloader()
        self.__train()
