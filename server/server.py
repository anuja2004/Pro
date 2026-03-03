import flwr as fl
from typing import List
import numpy as np

def main():
    print("🚀 Federated Server Started")

    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=2,
        min_available_clients=2,
        on_fit_config_fn=lambda r: {"round": r},
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
