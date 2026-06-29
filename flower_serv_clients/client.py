# Purpose is for each bank to run on own machine, loads its data and trains model locallu
# and sends gradients back to Flwr server

import flwr as fl
import torch
import os
import pandas as pd
import torch.nn as nn
import numpy as np
from model import CreditScoringModel
#flwr is the federated learning framework, which handles comms between client and server, 
# flower handles sending gradients to server or receive an updated Model

#imported the creditscoring model to train locally on own data


# read which bank this client represents from environment variable, 
# auto sets to 'a' if nothing loaded
BANK_ID = os.environ.get('BANK_ID', 'a')
DATA_DIR = 'data/processed'

# load this bank's training and test data
X_train = pd.read_parquet(f'{DATA_DIR}/bank_{BANK_ID}_X_train.parquet')
X_test  = pd.read_parquet(f'{DATA_DIR}/bank_{BANK_ID}_X_test.parquet')
y_train = pd.read_parquet(f'{DATA_DIR}/bank_{BANK_ID}_y_train.parquet')
y_test  = pd.read_parquet(f'{DATA_DIR}/bank_{BANK_ID}_y_test.parquet')


#Since pytorch cant directly work with DataFrames from pandas, tensor can be uswed for data formating
# convert dataframes to PyTorch tensors — the format PyTorch needs for training
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
X_test_tensor  = torch.tensor(X_test.values,  dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
y_test_tensor  = torch.tensor(y_test.values,  dtype=torch.float32)

# wrap in DataLoader — handles batching and shuffling automatically
train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

class CreditClient(fl.client.NumPyClient):
    def __init__(self):
        input_dim = X_train.shape[1]

        self.model = CreditScoringModel(input_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.BCELoss()

    def get_parameters(self, config):
        # state_dict() returns every weight and bias in the model as a dictionary
        # .values() pulls out just the tensors, not their names
        # .cpu() makes sure tensors aren't stuck on a GPU before converting
        # .numpy() converts each tensor into a numpy array — the format Flower needs to send over the network
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        # pair up the model's layer names with the incoming weight values from the server
        params_dict = zip(self.model.state_dict().keys(), parameters)

        # convert each incoming numpy array back into a PyTorch tensor
        # build a dictionary in the exact format PyTorch expects for loading weights
        state_dict = {k: torch.tensor(v) for k, v in params_dict}

        # actually load these weights into the model
        # strict=True throws an error if shapes don't match — catches bugs early
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        # load the global model weights sent from the server
        self.set_parameters(parameters)

        # train the model locally for one epoch on this bank's data
        self.model.train()
        for X_batch, y_batch in train_loader:
            self.optimizer.zero_grad()
            predictions = self.model(X_batch)
            loss = self.criterion(predictions, y_batch)
            loss.backward()
            self.optimizer.step()

        # return updated weights, number of training samples, and metrics
        return self.get_parameters(config={}), len(X_train), {}
    
    def evaluate(self, parameters, config):
        # load the global model weights sent from the server
        self.set_parameters(parameters)

        # switch to evaluation mode — turns off training-specific behavior
        self.model.eval()

        with torch.no_grad():
            predictions = self.model(X_test_tensor)
            loss = self.criterion(predictions, y_test_tensor).item()

            # convert probabilities to binary predictions using 0.5 threshold
            predicted_labels = (predictions >= 0.5).float()
            accuracy = (predicted_labels == y_test_tensor).float().mean().item()

        return loss, len(X_test), {"accuracy": accuracy}


if __name__ == "__main__":
    print(f"Starting client for Bank {BANK_ID.upper()}...")
    
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=CreditClient().to_client(),
    )