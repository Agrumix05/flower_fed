
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

import torch
import torch.nn as nn


###################
# ENCODER
###################
class Encoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=embedding_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        self.attn = nn.Linear(embedding_dim, 1)

        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):

        x = x.permute(0, 2, 1)

        output, _ = self.lstm(x)

        # attention pooling
        weights = torch.softmax(self.attn(output), dim=1)

        z = (weights * output).sum(dim=1)

        return self.norm(z)
###################
# DECODER
###################
class Decoder(nn.Module):
    def __init__(self, seq_len, embedding_dim=64, n_features=3):
        super().__init__()

        self.seq_len = seq_len
        self.embedding_dim = embedding_dim

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=embedding_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        self.output_layer = nn.Linear(embedding_dim, n_features)

    def forward(self, z):

        batch_size = z.size(0)

        # input iniziale nullo
        decoder_input = torch.zeros(
            batch_size,
            self.seq_len,
            self.output_layer.out_features,
            device=z.device
        )

        # usa embedding come hidden state iniziale
        h0 = z.unsqueeze(0).repeat(2, 1, 1)

        c0 = torch.zeros_like(h0)

        out, _ = self.lstm(decoder_input, (h0, c0))

        out = self.output_layer(out)

        return out.permute(0, 2, 1)
###################
# AUTOENCODER
###################
class RecurrentAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=64):
        super().__init__()

        self.encoder = Encoder(seq_len, n_features, embedding_dim)
        self.decoder = Decoder(seq_len, embedding_dim, n_features)

    def forward(self, x, return_embedding=False):
        z = self.encoder(x)
        x_hat = self.decoder(z)

        if return_embedding:
            return x_hat, z
        return x_hat
    

# =========================
# PREPROCESSING
# =========================

columns = [
    "heart_rate_data_time_series",
    "respiration_data_time_series",
    "stress_data_time_series",
]


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
    print(f"Loading data for partition {partition_id}...")
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
    all_rows = [pad_to_length(x, 664) for x in all_rows]
    all_rows = np.array(all_rows, dtype=np.float32)
    all_rows = np.nan_to_num(all_rows, nan=0.0)

    # torch tensor
    X = torch.tensor(all_rows, dtype=torch.float32)

    # NORMALIZZAZIONE LOCALE 
    X = (X - X.mean(dim=(0, 2), keepdim=True)) / (X.std(dim=(0, 2), keepdim=True) + 1e-8)
    dataset = TensorDataset(X)

    # split train/val
    n = len(dataset)
    train_size = int(0.8 * n)
    val_size = n - train_size

    train_set, val_set = random_split(dataset, [train_size, val_size])

    trainloader = DataLoader(train_set, batch_size=32, shuffle=True)
    valloader = DataLoader(val_set, batch_size=32)

    return trainloader, valloader








