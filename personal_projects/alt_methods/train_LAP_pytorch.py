import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from neural_network_LAP_pytorch import NeuralNetworkLAP

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# Hyperparameters
input_size = 32 * 32 * 3
hidden_size = 128
num_classes = 10
num_epochs = 10
batch_size = 100
learning_rate = 0.01
gamma = 0.9  # Momentum term for LAP optimizer

# CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize images
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                          shuffle=True, num_workers=2)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                         shuffle=False, num_workers=2)

# Initialize the neural network
model = NeuralNetworkLAP(input_size, hidden_size, num_classes).to(device)

# Loss function
criterion = nn.CrossEntropyLoss()

# Standard optimizer for comparison (e.g., SGD)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Training loop with LAP optimizer
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        images = images.view(-1, 32 * 32 * 3).to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        model.zero_grad()
        loss.backward()

        # Collect gradients
        grads = {
            'dW1': model.fc1.weight.grad,
            'db1': model.fc1.bias.grad,
            'dW2': model.fc2.weight.grad,
            'db2': model.fc2.bias.grad
        }

        # Update parameters using LAP optimizer
        model.least_action_optimizer(grads, alpha=learning_rate, gamma=gamma)

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item():.4f}')

# Evaluate the model on the test set
model.eval()  # Set model to evaluation mode
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.view(-1, 32 * 32 * 3).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Test Accuracy of the model on the 10000 test images: {100 * correct / total:.2f}%')
