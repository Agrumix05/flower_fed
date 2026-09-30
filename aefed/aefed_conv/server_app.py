# import flwr as fl
# from task import ConvAutoencoder

# # =========================
# # WEIGHTS UTILS
# # =========================

# def get_weights(net):
#     state_dict = net.state_dict()
#     return [state_dict[k].detach().cpu().numpy() for k in state_dict.keys()]


# # =========================
# # SERVER
# # =========================

# def main():

#     model = ConvAutoencoder(input_length=664, embedding_dim=32)

#     parameters = fl.common.ndarrays_to_parameters(get_weights(model))

#     strategy = fl.server.strategy.FedAvg(
#         fraction_fit=1.0,
#         fraction_evaluate=1.0,
#         min_available_clients=2,
#         initial_parameters=parameters,
#     )

#     fl.server.start_server(
#         server_address="127.0.0.1:8090",
#         config=fl.server.ServerConfig(num_rounds=40),
#         strategy=strategy,
#     )


# if __name__ == "__main__":
#     main()















# import flwr as fl
# import torch



# import flwr as fl
# import torch
# from task import ConvAutoencoder
# from flwr.common import parameters_to_ndarrays
# # =========================
# # WEIGHTS UTILS
# # =========================

# def get_weights(net):
#     state_dict = net.state_dict()
#     return [state_dict[k].detach().cpu().numpy() for k in state_dict.keys()]


# def set_weights(model, parameters):
#     state_dict = model.state_dict()
#     for k, v in zip(state_dict.keys(), parameters):
#         state_dict[k] = torch.tensor(v)
#     model.load_state_dict(state_dict)


# def save_model(model, ndarrays, path="final_model.pth"):
#     set_weights(model, ndarrays)
#     torch.save(model.state_dict(), path)
#     print(f"[INFO] Model saved to {path}")


# # =========================
# # SERVER
# # =========================

# def main():

#     model = ConvAutoencoder(input_length=664, embedding_dim=32)

#     initial_parameters = fl.common.ndarrays_to_parameters(
#         get_weights(model)
#     )

#     # =========================
#     # STRATEGY
#     # =========================

#     class SaveModelFedAvg(fl.server.strategy.FedAvg):
#         def __init__(self, model, *args, **kwargs):
#             super().__init__(*args, **kwargs)
#             self.model = model

#         def aggregate_fit(self, server_round, results, failures):

#             aggregated = super().aggregate_fit(server_round, results, failures)

#             if aggregated is None:
#                 return None

#             parameters_aggregated, metrics = aggregated
#             ndarrays = parameters_to_ndarrays(parameters_aggregated)

#             if server_round == 40:
#                 save_model(self.model, ndarrays)

#             return aggregated


#     strategy = SaveModelFedAvg(
#         model=model,
#         fraction_fit=1.0,
#         fraction_evaluate=1.0,
#         min_available_clients=2,
#         initial_parameters=initial_parameters,
#     )

#     fl.server.start_server(
#         server_address="127.0.0.1:8090",
#         config=fl.server.ServerConfig(num_rounds=100),
#         strategy=strategy,
#     )


# if __name__ == "__main__":
#     main()




import flwr as fl
import torch

from task import ConvAutoencoder
from flwr.common import parameters_to_ndarrays


# =========================
# WEIGHTS UTILS
# =========================

def get_weights(model):
    state_dict = model.state_dict()
    return [v.detach().cpu().numpy() for v in state_dict.values()]


def set_weights(model, ndarrays):
    state_dict = model.state_dict()
    for k, v in zip(state_dict.keys(), ndarrays):
        state_dict[k] = torch.tensor(v)
    model.load_state_dict(state_dict)


def save_model(model, ndarrays, path="best_model.pth"):
    set_weights(model, ndarrays)
    torch.save(model.state_dict(), path)
    print(f"[INFO] Saved model to {path}")


# =========================
# STRATEGY WITH EARLY STOPPING
# =========================

class FedAvgEarlyStopping(fl.server.strategy.FedAvg):
    def __init__(
        self,
        model,
        patience=5,
        min_delta=1e-4,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.model = model
        self.patience = patience
        self.min_delta = min_delta

        self.best_loss = float("inf")
        self.bad_rounds = 0
        self.should_stop = False

        self.best_params = None

    # -------------------------
    # TRAIN AGGREGATION
    # -------------------------
    def aggregate_fit(self, server_round, results, failures):
        return super().aggregate_fit(server_round, results, failures)

    # -------------------------
    # VALIDATION AGGREGATION
    # -------------------------
    def aggregate_evaluate(self, server_round, results, failures):
        aggregated = super().aggregate_evaluate(server_round, results, failures)

        if aggregated is None:
            return None

        loss, metrics = aggregated

        print(f"[Round {server_round}] val_loss={loss:.6f}")

        # ---- EARLY STOPPING LOGIC ----
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.bad_rounds = 0

            # salva best parameters
            if self.current_parameters is not None:
                self.best_params = self.current_parameters

        else:
            self.bad_rounds += 1

        print(f"bad_rounds: {self.bad_rounds}/{self.patience}")

        if self.bad_rounds >= self.patience:
            print("[INFO] Early stopping triggered.")
            self.should_stop = True

            # salva modello finale migliore
            if self.best_params is not None:
                ndarrays = parameters_to_ndarrays(self.best_params)
                save_model(self.model, ndarrays, "best_model.pth")

        return aggregated

    # -------------------------
    # STOP TRAINING CLIENTS
    # -------------------------
    def configure_fit(self, server_round, parameters, client_manager):
        if self.should_stop:
            return []
        self.current_parameters = parameters
        return super().configure_fit(server_round, parameters, client_manager)

    def configure_evaluate(self, server_round, parameters, client_manager):
        if self.should_stop:
            return []
        return super().configure_evaluate(server_round, parameters, client_manager)


# =========================
# SERVER MAIN
# =========================

def main():

    model = ConvAutoencoder(input_length=700, embedding_dim=16)

    initial_parameters = fl.common.ndarrays_to_parameters(
        get_weights(model)
    )

    strategy = FedAvgEarlyStopping(
        model=model,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=initial_parameters,
        patience=5,
        min_delta=1e-4,
    )

    fl.server.start_server(
        server_address="127.0.0.1:8090",
        config=fl.server.ServerConfig(num_rounds=3000),  # alto, ma verrà fermato prima
        strategy=strategy,
    )


if __name__ == "__main__":
    main()