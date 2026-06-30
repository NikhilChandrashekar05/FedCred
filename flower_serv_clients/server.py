# imports our custom strategy that extends FedAvg with SQLite logging
from strategy import FedCreditStrategy
import flwr as fl

# how many federated learning rounds to run
NUM_ROUNDS = 10

# minimum number of banks that must be connected before training starts
MIN_CLIENTS = 3

def main():
    # use our custom strategy instead of plain FedAvg
    # this version logs accuracy and loss to SQLite after every round
    strategy = FedCreditStrategy(
        # server waits until all 3 banks are connected before starting round 1
        min_available_clients=MIN_CLIENTS,

        # use all connected banks for training each round — 1.0 means 100%
        fraction_fit=1.0,

        # use all connected banks for evaluation each round
        fraction_evaluate=1.0,
    )

    # start the Flower server and begin listening for clients on port 8080
    # once all 3 clients connect it automatically kicks off round 1
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

if __name__ == "__main__":
    print(f"Starting Flower server — waiting for {MIN_CLIENTS} clients...")
    main()