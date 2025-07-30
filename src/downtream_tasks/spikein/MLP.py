import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    Multi Layer Preceptron for Spike-in 
    """

    def __init__(self, in_dim, h_dim, out_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.dp1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(h_dim, h_dim//2)
        self.dp2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(h_dim//2, out_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dp1(x)
        x = torch.relu(self.fc2(x))
        x = self.dp2(x)
        x = self.fc3(x)
        return x