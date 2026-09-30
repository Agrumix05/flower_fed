import sys
import torch
import torch.nn as nn
import torch.optim as optim
import flwr as fl

# =========================
# FLOWER CLIENT
# =========================
from task import ConvAutoencoder, load_data


class FlowerClient(fl.client.NumPyClient):

    def __init__(self, partition_id):

        print("Client ID:", partition_id)

        self.net = ConvAutoencoder(
            input_length=700,
            embedding_dim=16
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

        self.trainloader, self.valloader = load_data(partition_id)

    # -------------------------
    # GET / SET PARAMS
    # -------------------------

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.net.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = self.net.state_dict()
        keys = list(state_dict.keys())

        new_state_dict = {}

        for k, v in zip(keys, parameters):
            # convert numpy -> torch tensor
            new_state_dict[k] = torch.from_numpy(v)

        self.net.load_state_dict(new_state_dict, strict=True)
    # -------------------------
    # TRAIN (FIT)
    # -------------------------

    def fit(self, parameters, config):

        self.set_parameters(parameters)

        self.net.train()
        optimizer = optim.Adam(self.net.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        total_loss = 0.0

        for _ in range(1):  # 1 epoch per round (standard FL)
            for batch in self.trainloader:
                x = batch[0].to(self.device)

                optimizer.zero_grad()
                recon = self.net(x)

                loss = criterion(recon, x)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

        avg_loss = total_loss / len(self.trainloader)

        return self.get_parameters(config), len(self.trainloader.dataset), {"loss": avg_loss}

    # -------------------------
    # EVALUATION
    # -------------------------

    def evaluate(self, parameters, config):

        self.set_parameters(parameters)

        self.net.eval()
        criterion = nn.MSELoss()

        total_loss = 0.0

        with torch.no_grad():
            for batch in self.valloader:
                x = batch[0].to(self.device)
                recon = self.net(x)
                loss = criterion(recon, x)
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

