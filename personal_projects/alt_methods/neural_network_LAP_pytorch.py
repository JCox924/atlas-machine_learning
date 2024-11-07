import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralNetworkLAP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetworkLAP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

        # Initialize velocities for LAP optimizer
        self.VdW1 = torch.zeros_like(self.fc1.weight)
        self.Vdb1 = torch.zeros_like(self.fc1.bias)
        self.VdW2 = torch.zeros_like(self.fc2.weight)
        self.Vdb2 = torch.zeros_like(self.fc2.bias)

    def forward(self, x):
        out = torch.tanh(self.fc1(x))
        out = self.fc2(out)
        return out  # We'll apply softmax in the loss function

    def least_action_optimizer(self, grads, alpha=0.01, gamma=0.9):
        # Update velocities
        self.VdW1 = gamma * self.VdW1 + alpha * grads['dW1']
        self.Vdb1 = gamma * self.Vdb1 + alpha * grads['db1']
        self.VdW2 = gamma * self.VdW2 + alpha * grads['dW2']
        self.Vdb2 = gamma * self.Vdb2 + alpha * grads['db2']

        # Update parameters
        with torch.no_grad():
            self.fc1.weight -= self.VdW1
            self.fc1.bias -= self.Vdb1
            self.fc2.weight -= self.VdW2
            self.fc2.bias -= self.Vdb2
