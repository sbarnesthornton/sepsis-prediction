import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torch.nn.utils.rnn import pack_sequence
from sklearn.metrics import f1_score, roc_auc_score

import torch.optim as optim
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import precision_recall_fscore_support as score
import csv

from uscore import Uscore


def write_to_csv(y_test, y_pred, meta, filename):
    test_values = y_test.tolist()
    pred_values = y_pred.tolist()
    print(meta)
    patient_ids = meta[["Patient_ID"]].to_numpy().flatten()
    with open(filename, "w+", newline="") as csvfile:
        field_names = ["logit", "ground_truth", "patient_id"]
        writer = csv.DictWriter(csvfile, fieldnames=field_names)

        writer.writeheader()
        for i in range(len(test_values)):
            writer.writerow(
                {
                    "logit": pred_values[i],
                    "ground_truth": test_values[i],
                    "patient_id": patient_ids[i],
                }
            )


class VitalSignsDataset(Dataset):
    """
    A dataset that returns windowed time-series snapshots of a patient
    """

    def __init__(self, path, slice_size):
        """
        Parameters
        ----------
        path: str
            path to csv file
        slice_size: int
            size of window of patient data
        """

        self.df = pd.read_csv(path)
        self.slice_size = slice_size

        index_conversion = self.df[
            self.df.groupby("Patient_ID").cumcount(ascending=False)
            > self.slice_size - 1
        ]
        index_conversion = index_conversion.reset_index()
        self.idx_to_idx = index_conversion[["index"]].to_numpy().reshape(-1)
        meta_cols = [
            "Patient_ID",
            "SepsisLabel",
            "ICULOS",
            "HospAdmTime",
            "Gender",
            "Age",
            "Hour",
        ]
        self.vitals = self.df.drop(
            meta_cols,
            axis=1,
        ).to_numpy(np.float32)
        self.labels = self.df[["SepsisLabel"]].to_numpy(np.float32)
        self.meta = self.df[meta_cols].to_numpy(np.float32)
        self.static = self.df[["Gender", "Age"]].to_numpy(np.float32)

    def __len__(self):
        return len(self.idx_to_idx)

    def __getitem__(self, idx):
        index = self.idx_to_idx[idx]
        return (
            self.vitals[(index) : (index + self.slice_size)],
            self.labels[index + self.slice_size - 1],
            self.meta[index + self.slice_size - 1],
            self.static[index],
        )

    def get_labels(self):
        return [0, 1]


class StartDataset(Dataset):
    """
    A dataset that returns a single window snapshot of each patient at the
    start of their stay
    """

    def __init__(self, annotations, slice_size):
        """
        Parameters
        ----------
        path: str
            path to csv file
        slice_size: int
            size of window of patient data
        """

        self.df = pd.read_csv(annotations)
        self.slice_size = slice_size

        # remove patients with fewer entries than slice size
        df_wo_shortstay = self.df[
            self.df.groupby("Patient_ID").cumcount(ascending=False)
            > self.slice_size - 1
        ]

        # convert between index over data to index over full dataframe
        indices = df_wo_shortstay.reset_index().groupby("Patient_ID").first()["index"]
        self.indices = indices.values

        # columns to be removed from training data
        meta_cols = [
            "Patient_ID",
            "SepsisLabel",
            "ICULOS",
            "HospAdmTime",
            "Gender",
            "Age",
        ]
        self.vitals = self.df.drop(
            meta_cols,
            axis=1,
        ).to_numpy(np.float32)

        # whether each patient had sepsis at the end of their stay
        self.labels = (
            self.df.reset_index()
            .groupby("Patient_ID")
            .last()["SepsisLabel"]
            .to_numpy(np.float32)
            .reshape((-1, 1))
        )

        # patient metadata at start of their stay
        self.meta = (
            self.df.reset_index()
            .groupby("Patient_ID")
            .first()
            .reset_index()[meta_cols]
            .to_numpy(np.float32)
        )

        self.static = (
            self.df.reset_index()
            .groupby("Patient_ID")
            .last()[["Gender", "Age"]]
            .to_numpy(np.float32)
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        index = self.indices[idx]
        return (
            self.vitals[(index) : (index + self.slice_size)],
            self.labels[idx],
            self.meta[idx],
            self.static[idx],
        )

    def get_labels(self):
        return [0, 1]


class TimeSeriesModel(nn.Module):
    """
    A basic LSTM model for prediction of sepsis
    """

    def __init__(self, input_size, hidden_size=20):
        """
        Parameters
        ----------
        input_size: int
            number of dimensions for the input data
        hidden_size: int
            number of dimensions for the hidden state of LSTM cells
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=40,
            num_layers=4,
            batch_first=True,
            dropout=0.2,
        )
        self.fc1 = nn.Linear(40, 1)

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        output = self.fc1(output[:, -1])
        return output


class TimeSeriesModelConv(nn.Module):
    """
    A model that combines a convolution layer with an LSTM
    """

    def __init__(
        self, input_size, hidden_size=40, lstm_hidden_size=20, static_size=2, lrf=4
    ):
        """
        Parameters
        ----------
        input_size: int
            number of dimensions for the input data
        hidden_size: int
            number of channels to be produced by the convolution layer
        lstm_hidden_size: int
            number of dimensions for the hidden state of LSTM cells
        static_size: int
            number of dimensions for static patient values
        lrf: int
            number of time series steps to be spanned by the convolution kernel
        """
        super().__init__()
        self.conv = nn.Conv2d(1, hidden_size, kernel_size=(lrf, input_size))
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=4,
            batch_first=True,
            dropout=0.2,
        )
        self.fc1 = nn.Linear(lstm_hidden_size + static_size, 1)

    def forward(self, x, static):
        x = x.unsqueeze(dim=1)
        x = self.conv(x)
        x = self.relu(x)
        x = x.squeeze()
        x = torch.permute(x, (0, 2, 1))
        output, (h_n, c_n) = self.lstm(x)
        full_output = torch.cat((output[:, -1], static), dim=1)
        output = self.fc1(full_output)
        return output


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

input_cols = [
    "HR",
    "O2Sat",
    "Temp",
    "SBP",
    "MAP",
    "DBP",
    "Resp",
    "BaseExcess",
    "HCO3",
    "FiO2",
    "pH",
    "PaCO2",
    "SaO2",
    "AST",
    "BUN",
    "Alkalinephos",
    "Calcium",
    "Chloride",
    "Creatinine",
    "Glucose",
    "Lactate",
    "Magnesium",
    "Phosphate",
    "Potassium",
    "Bilirubin_total",
    "Hct",
    "Hgb",
    "PTT",
    "WBC",
    "Platelets",
]
sepsis_cols = ["SepsisLabel"]
meta_cols = [
    "Patient_ID",
    "SepsisLabel",
    "ICULOS",
    "HospAdmTime",
    "Gender",
    "Age",
    "Hour",
]


train_dataset = VitalSignsDataset(
    path="data/train_set_interpolation_with_multivariate.csv", slice_size=16
)
val_dataset = VitalSignsDataset(
    path="data/val_set_interpolation_with_multivariate.csv", slice_size=16
)
test_dataset = VitalSignsDataset(
    path="data/test_set_interpolation_with_multivariate.csv", slice_size=16
)

labels = [int(label) for _, label, _, _ in train_dataset]
class_weights = {"0": 0.023471489401092338, "1": 0.9765285105989077}
sample_weights = [class_weights[str(label)] for label in labels]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

train_loader = DataLoader(train_dataset, batch_size=1024, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=len(val_dataset))
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))


model = TimeSeriesModelConv(input_size=30)
model.to(device)
print(model)

optimizer = optim.Adam(
    model.parameters(), lr=5e-4, betas=(0.9, 0.999), weight_decay=1e-5
)
loss_fn = nn.BCEWithLogitsLoss()


n_epochs = 50
for epoch in range(n_epochs):
    model.train()
    for i, (X_batch, y_batch, meta, static) in enumerate(train_loader):
        X_batch = X_batch.to(device)
        static = static.to(device)
        y_batch = y_batch.to(device)

        y_pred = model(X_batch, static)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            preds = (
                torch.sigmoid(y_pred.cpu().flatten()) > torch.Tensor([0.5])
            ).float()
            train_f1 = f1_score(y_batch.cpu().flatten(), preds)
            print(
                f"Epoch {epoch}, iteration {i}, loss: {loss.item():.4f}, f1 score: {f1_score(y_batch.cpu().flatten(), preds,average= None)}, ROC AUC {roc_auc_score(y_batch.cpu().flatten(), preds)}"
            )
    # Validation
    if epoch % 5 != 0:
        continue
    model.eval()
    with torch.no_grad():
        preds = []
        for i, (X_test, y_test, meta, static) in enumerate(val_loader):
            X_test = X_test.to(device)
            static = static.to(device)
            y_test = y_test.to(device)
            y_pred = model(X_test, static)
            preds = (
                torch.sigmoid(y_pred.cpu().flatten()) > torch.Tensor([0.5])
            ).float()
            train_f1 = f1_score(y_test.cpu().flatten(), preds)
            print(
                f"Validation loss: {loss.item():.4f}, test f1 score: {f1_score(y_test.cpu().flatten(), preds,average= None)}, test ROC AUC {roc_auc_score(y_test.cpu().flatten(), preds)}"
            )
            X_data = X_test.cpu().numpy()[:, -1, :]
            X_meta = meta.cpu().numpy()
            Y_actual = y_test.cpu().numpy().flatten()
            Y_preds = preds.cpu().numpy().flatten()
            X_meta_df = pd.DataFrame(
                data=X_meta, index=np.arange(0, X_meta.shape[0]), columns=meta_cols
            )
            Y_actual_series = pd.Series(
                data=Y_actual, index=np.arange(0, Y_actual.shape[0]), name="SepsisLabel"
            )
            Y_preds_series = pd.Series(
                data=Y_preds, index=np.arange(0, Y_actual.shape[0]), name="SepsisLabel"
            )
            uscore = Uscore(Y_preds, Y_actual_series, X_meta_df)
            print(f"UScore: {uscore}")
    if epoch == n_epochs - 1:
        with torch.no_grad():
            preds = []
            for i, (X_test, y_test, meta, static) in enumerate(val_loader):
                X_test = X_test.to(device)
                static = static.to(device)
                y_test = y_test.to(device)
                y_pred = model(X_test, static)
                preds = (
                    torch.sigmoid(y_pred.cpu().flatten()) > torch.Tensor([0.5])
                ).float()
                train_f1 = f1_score(y_test.cpu().flatten(), preds)
                print(
                    f"Test loss: {loss.item():.4f}, test f1 score: {f1_score(y_test.cpu().flatten(), preds,average= None)}, test ROC AUC {roc_auc_score(y_test.cpu().flatten(), preds)}"
                )
                X_data = X_test.cpu().numpy()[:, -1, :]
                X_meta = meta.cpu().numpy()
                Y_actual = y_test.cpu().numpy().flatten()
                Y_preds = preds.cpu().numpy().flatten()
                X_meta_df = pd.DataFrame(
                    data=X_meta, index=np.arange(0, X_meta.shape[0]), columns=meta_cols
                )
                Y_actual_series = pd.Series(
                    data=Y_actual,
                    index=np.arange(0, Y_actual.shape[0]),
                    name="SepsisLabel",
                )
                Y_preds_series = pd.Series(
                    data=Y_preds,
                    index=np.arange(0, Y_actual.shape[0]),
                    name="SepsisLabel",
                )
                uscore = Uscore(Y_preds, Y_actual_series, X_meta_df)
                print(f"UScore: {uscore}")
                write_to_csv(
                    y_test.cpu().numpy().flatten(),
                    torch.sigmoid(y_pred.cpu().flatten()).numpy(),
                    X_meta_df,
                    "./probs.csv",
                )
