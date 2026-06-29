#Server.py is the central aggregator, works with all clients runs FedAvg
import flwr as fl
#Client loads data, model builds, trains and evaluate

# number of federated learning rounds to run
NUM_ROUNDS = 10

# minimum number of clients that must be connected before training starts
MIN_CLIENTS = 3

def main():
    # define the federated learning strategy
    strategy = fl.server.strategy.FedAvg(
        # wait for all 3 clients before starting each round
        min_available_clients=MIN_CLIENTS,
        
        # use all connected clients for training each round
        fraction_fit=1.0,
        
        # use all connected clients for evaluation each round
        fraction_evaluate=1.0,
    )

    # start the flower server
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

if __name__ == "__main__":
    print(f"Starting Flower server, waiting for {MIN_CLIENTS} clients...")
    main()