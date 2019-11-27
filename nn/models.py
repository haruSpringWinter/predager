import torch.nn as nn
import torch

class MLP(nn.Module):
    def __init__(self, input_dim, num_hidden, out_dim):
        self.input_dim = input_dim
        self.num_hidden = num_hidden
        self.out_dim = out_dim

        self.layers = nn.Sequential(
            nn.Linear(input_dim, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, out_dim)
        )
        
    def forward(self, x):
        x = self.layers(x)
        return x
