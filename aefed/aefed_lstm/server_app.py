import flwr as fl
import torch
from task import RecurrentAutoencoder
from flwr.common import parameters_to_ndarrays


# =========================
# WEIGHTS UTILS
# =========================

def get_weights(net):
    state_dict = net.state_dict()
    return [state_dict[k].detach().cpu().numpy() for k in state_dict.keys()]


def set_weights(net, parameters):
    state_dict = net.state_dict()
    new_state_dict = {}

    for k, v in zip(state_dict.keys(), parameters):
        new_state_dict[k] = torch.tensor(v)

    net.load_state_dict(new_state_dict, strict=True)


def save_model(model, parameters, path="final_model.pth"):
    set_weights(model, parameters)
    torch.save(model.state_dict(), path)
    print(f"[INFO] Model saved to {path}")


# =========================
# SERVER
# =========================

def main():

    model = RecurrentAutoencoder(
        seq_len=664,
        n_features=3,
        embedding_dim=32
    )

    initial_parameters = fl.common.ndarrays_to_parameters(
        get_weights(model)
    )

    class SaveFinalModel(fl.server.strategy.FedAvg):
        def __init__(self, model, num_rounds, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.model = model
            self.num_rounds = num_rounds

        def aggregate_fit(self, server_round, results, failures):
            aggregated = super().aggregate_fit(server_round, results, failures)

            if aggregated is None:
                return None

            parameters_aggregated, metrics = aggregated
            ndarrays = parameters_to_ndarrays(parameters_aggregated)

            # salva solo all’ultimo round
            if server_round == self.num_rounds:
                save_model(self.model, ndarrays)

            return aggregated


    strategy = SaveFinalModel(
        model=model,
        num_rounds=40,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=initial_parameters,
    )

    fl.server.start_server(
        server_address="127.0.0.1:8090",
        config=fl.server.ServerConfig(num_rounds=40),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()