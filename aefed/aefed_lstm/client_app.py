import sys
import torch
import torch.nn as nn
import torch.optim as optim
import flwr as fl

from torch.utils.data import DataLoader, TensorDataset

from task import RecurrentAutoencoder, load_data

# =========================
# FLOWER CLIENT
# =========================


class FlowerClient(fl.client.NumPyClient):

    def __init__(self, partition_id):

        print("Client ID:", partition_id)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net = RecurrentAutoencoder(
            seq_len=664,
            n_features=3,
            embedding_dim=32
            
        ).to(self.device)

        # =========================
        # DATA (già preprocessato come nel tuo centralizzato)
        # =========================
        self.trainloader, self.valloader = load_data(partition_id)

        # LOSS e OPTIMIZER verranno creati ogni round (come in FL standard)
        self.criterion = nn.MSELoss()

    # =========================
    # PARAMS
    # =========================

    def get_parameters(self, config=None):
        return [v.detach().cpu().numpy() for v in self.net.state_dict().values()]

    def set_parameters(self, parameters):

        state_dict = self.net.state_dict()
        new_state_dict = {}

        for k, v in zip(state_dict.keys(), parameters):
            new_state_dict[k] = torch.tensor(v, dtype=torch.float32)

        self.net.load_state_dict(new_state_dict, strict=True)

    # =========================
    # TRAIN (IDENTICO AL TUO LOOP CENTRALIZZATO)
    # =========================

    def fit(self, parameters, config):

        self.set_parameters(parameters)

        optimizer = optim.Adam(self.net.parameters(), lr=1e-3)

        self.net.train()

        num_epochs = 1  # FL = 1 epoch per round (standard)

        total_loss = 0.0

        for _ in range(num_epochs):

            for batch in self.trainloader:

                x = batch[0].to(self.device)  # (B, 3, 664)

                optimizer.zero_grad()

                recon = self.net(x)

                loss = self.criterion(recon, x)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

        avg_loss = total_loss / len(self.trainloader)

        return (
            self.get_parameters(),
            len(self.trainloader.dataset),
            {"loss": avg_loss},
        )

    # =========================
    # EVALUATION
    # =========================

    def evaluate(self, parameters, config):

        self.set_parameters(parameters)

        self.net.eval()

        total_loss = 0.0

        with torch.no_grad():

            for batch in self.valloader:

                x = batch[0].to(self.device)

                recon = self.net(x)

                loss = self.criterion(recon, x)

                total_loss += loss.item()

        avg_loss = total_loss / len(self.valloader)

        return float(avg_loss), len(self.valloader.dataset), {"loss": avg_loss}


# =========================
# START CLIENT
# =========================

if __name__ == "__main__":

    partition_id = int(sys.argv[1])

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8090",
        client=FlowerClient(partition_id),
    )