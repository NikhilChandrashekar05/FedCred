import flwr as fl
# importing the built-in FedAvg so we can subclass it and add logging on top
# we are not rebuilding FedAvg from scratch, just extending it
from flwr.server.strategy import FedAvg

from flwr.common import Scalar

# sqlite3 is Python's built in database library, no installation needed
# this is what saves round metrics like accuracy and loss to a local database file
# FastAPI will read from this database later to show the dashboard
import sqlite3
import os
# type hints just make the code cleaner and are standard practice
# Optional means a value could be None
# Union means a value could be one of several possible types
from typing import Optional, Union

# these are Flower's internal data types
# FitRes is the result that comes back after a client finishes training
# EvaluateRes is the result that comes back after a client finishes evaluating
# Parameters represents the model weights being passed around
from flwr.common import FitRes, EvaluateRes, Parameters
from flwr.server.client_proxy import ClientProxy

# path to the database file — created automatically if it doesn't exist
DB_PATH = 'api/fedcredit.db'


def init_db():
    # connect to the database, creating the file if it doesn't already exist
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # create a table to store metrics for each round, only if doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS rounds ( round_num INTEGER PRIMARY KEY, loss REAL, accuracy REAL, num_clients INTEGER)
    ''')

    conn.commit()
    conn.close()


def log_round(round_num: int, loss: float, accuracy: float, num_clients: int):
    # connect to the database file
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # insert this round's metrics into the rounds table
    # INSERT OR REPLACE means if this round number already exists, overwrite it
    # this prevents errors if you re-run training
    # the ? marks are placeholders — SQLite fills them in safely with the tuple values
    # never format values directly into a SQL string, it opens security vulnerabilities
    cursor.execute(
        'INSERT OR REPLACE INTO rounds (round_num, loss, accuracy, num_clients) VALUES (?, ?, ?, ?)',
        (round_num, loss, accuracy, num_clients)
    )

    # save the changes and close the connection cleanly
    conn.commit()
    conn.close()

    # confirm in the terminal that this round was saved successfully
    print(f"  [DB] Round {round_num} logged — loss: {loss:.4f} | accuracy: {accuracy:.4f} | clients: {num_clients}")

class FedCreditStrategy(FedAvg):
    def __init__(self, *args, **kwargs):
        # call FedAvg's own init first so it sets itself up properly
        # *args and **kwargs pass through whatever arguments the server gave us
        # without needing to know what they are
        super().__init__(*args, **kwargs)
        
        # create the api/ folder if it doesn't exist yet
        # DB_PATH is 'api/fedcredit.db' so the api/ folder must exist first
        # exist_ok=True means don't throw an error if the folder is already there
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        
        # create the database and rounds table before training starts
        # this runs once at startup, not every round
        init_db()
        
        print("FedCredStrategy initialized, the database ready")


    def aggregate_fit(self, server_round, results, failures):
        # let FedAvg do its averaging math first — we never touch this part
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        # if the round produced a valid model, log it to the database
        if aggregated_parameters is not None:

            # how many banks participated this round
            num_clients = len(results)

            # get the loss each bank reported and average them
            losses = [fit_res.metrics.get('loss', 0) for _, fit_res in results]
            avg_loss = sum(losses) / len(losses) if losses else 0.0

            # save to SQLite — FastAPI reads this for the dashboard
            log_round(server_round, avg_loss, 0.0, num_clients)

        # give Flower the updated model so it can send it to clients next round
        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(self, server_round, results, failures):
        # let FedAvg handle the evaluation aggregation first
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )

        # if we got valid results back from the clients
        if results:
            # get the accuracy each bank reported on their test data
            accuracies = [evaluate_res.metrics.get('accuracy', 0) for _, evaluate_res in results]
            
            # average the accuracies across all banks
            avg_accuracy = sum(accuracies) / len(accuracies)

            # update this round's database entry with the real accuracy
            log_round(server_round, aggregated_loss or 0.0, avg_accuracy, len(results))

            print(f"  Round {server_round} : avg accuracy across banks: {avg_accuracy:.4f}")

        # give Flower the aggregated results so it can continue
        return aggregated_loss, aggregated_metrics

if __name__ == "__main__":
    print("Starting FedCredit server with logging strategy")

    # use our custom strategy instead of plain FedAvg
    # everything else stays the same as before
    strategy = FedCreditStrategy(
        # wait for all 3 banks before starting each round
        min_available_clients=3,

        # use all connected banks for training each round
        fraction_fit=1.0,

        # use all connected banks for evaluation each round
        fraction_evaluate=1.0,
    )

    # start the flower server with our custom strategy
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )