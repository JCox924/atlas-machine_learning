import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Define neural network architecture
class NeuralNetwork(nn.Module):
    def __init__(self, nx, nodes, classes):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(nx, nodes)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(nodes, classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# Move model to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNetwork(nx=3072, nodes=128, classes=10).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)