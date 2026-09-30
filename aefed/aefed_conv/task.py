
import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import ast

from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader, random_split


# =========================
# MODELLO AUTOENCODER
# =========================

class ConvAutoencoder(nn.Module):
    def __init__(self, input_length=664, embedding_dim=32):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.flatten = nn.Flatten()

        self.fc_enc = nn.Linear(32 * (input_length // 4), embedding_dim)
        self.fc_dec = nn.Linear(embedding_dim, 32 * (input_length // 4))

        self.unflatten = nn.Unflatten(1, (32, input_length // 4))

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 3, kernel_size=2, stride=2),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)

        z = self.fc_enc(x)

        x = self.fc_dec(z)
        x = self.unflatten(x)

        x = self.decoder(x)
        return x


# =========================
# PREPROCESSING
# =========================

columns = [
    "heart_rate_data_time_series",
    "respiration_data_time_series",
    "stress_data_time_series",
]

# columns = [
#     "sleep_data_respiration_time_series",
#     "sleep_data_heart_rate_time_series",
#     "sleep_data_stress_time_series",
# ]

def parse_ts(ts_string):
    if pd.isna(ts_string):
        return None
    if isinstance(ts_string, str) and ts_string.strip() == "":
        return None
    try:
        data = ast.literal_eval(ts_string)
        t = np.array([p[0] for p in data], dtype=np.float32)
        v = np.array([p[1] for p in data], dtype=np.float32)
        return t, v
    except:
        return None



def align_series(ts_list):
    # unione di tutti i timestamp reali
    all_t = np.concatenate([t for t, _ in ts_list])
    t_common = np.unique(all_t)  
    # Rimuove i timestamp duplicati 
    aligned = []

    for t, v in ts_list:
        # interp SOLO dentro il range reale
        v_interp = np.interp(t_common, t, v, left=np.nan, right=np.nan)
        aligned.append(v_interp)

    aligned = np.stack(aligned, axis=0)  # (3, len_t)
    return aligned



def pad_to_length(arr, target_len):
    if arr.shape[1] >= target_len:
        return arr
    pad = np.zeros((arr.shape[0], target_len - arr.shape[1]), dtype=np.float32)
    return np.concatenate([arr, pad], axis=1)


# =========================
# DATA LOADER PER CLIENT
# =========================

def load_data(partition_id: int):

    folder = Path(f"./data/{partition_id}")

    all_rows = []
    max_len = 0

    for file in folder.glob("*.csv"):
        df = pd.read_csv(file)

        for _, row in df.iterrows():

            ts_list = []
            valid = True

            for col in columns:
                parsed = parse_ts(row[col])
                if parsed is None or len(parsed[0]) == 0:
                    valid = False
                    break
                ts_list.append(parsed)

            if not valid:
                continue

            aligned = align_series(ts_list)

            mask = ~np.all(np.isnan(aligned), axis=0)
            aligned = aligned[:, mask]

            if aligned.shape[1] == 0:
                continue

            max_len = max(max_len, aligned.shape[1])
            all_rows.append(aligned)

    # padding
    all_rows = [pad_to_length(x, 700) for x in all_rows]
    all_rows = np.array(all_rows, dtype=np.float32)
    all_rows = np.nan_to_num(all_rows, nan=0.0)

    # torch tensor
    X = torch.tensor(all_rows, dtype=torch.float32)

    # NORMALIZZAZIONE LOCALE 
    X = (X - X.mean(dim=(0, 2), keepdim=True)) / (X.std(dim=(0, 2), keepdim=True) + 1e-8)
    dataset = TensorDataset(X)

    # split train/val
    n = len(dataset)



    '''  NOTE: change fraction to split train/val (e.g., 0.8 for 80% train, 20% val) '''
    fraction = 0.8
    train_size = int(fraction * n)
    val_size = n - train_size

    train_set, val_set = random_split(dataset, [train_size, val_size])

    trainloader = DataLoader(train_set, batch_size=32, shuffle=True)
    valloader = DataLoader(val_set, batch_size=32)

    return trainloader, valloader
