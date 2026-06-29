# Contains the actual neural network that each bank trains locally
import torch #ML framework that I am using to build and train NN
import torch.nn as nn #type of sub aspect of PyTorch that has the template on N.N 

#class that inherits from PyTorch nural network class
class CreditScoringModel(nn.Module):

    def __init__(self, input_dim=16):
        super(CreditScoringModel, self).__init__()
        #.Sequential stacks layers in order data flows top to bottom
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64,32),
            nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid() 
        )
        #Model starts pattern learning from turing 16->64 values 
        #nn ReLU() takes negative value sets to 0 and keeps positive values same
        # efficient becuz it helps learn complex patterns
        # nn.Linear(64, 32) compresses the 64 vals to 32 extracts important signals
        # then takes 32 -> 1 which is the final prediction
        # nn.Sigmoid() squishes the number into range between 0-1 which turns to %

    def forward(self, x):
        #pass inputs through network and return prediciton
        return self.network(x)