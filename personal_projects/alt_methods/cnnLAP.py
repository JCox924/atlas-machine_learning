import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class LAPCNN(nn.Module):
    def __init__(self):
        super(LAPCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)

        # Initialize velocities for LAP
        self.init_velocities()

    def init_velocities(self):
        self.velocities = {}
        for name, param in self.named_parameters():
            self.velocities[name] = torch.zeros_like(param.data)

    def forward(self, x):
        # Convolutional layers with max pooling
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)

        # Flatten and fully connected layers
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x




class LAPOptimizerCNN:
    def __init__(self, model, alpha=0.01, gamma=0.9):
        """
        LAP Optimizer adapted for CNNs
        args:
            model: The neural network model
            alpha: Learning rate
            gamma: Momentum coefficient
        """
        self.model = model
        self.alpha = alpha
        self.gamma = gamma

        # Initialize adaptive parameters
        self.layer_scales = defaultdict(lambda: 1.0)
        self.grad_history = defaultdict(list)
        self.max_history = 50

    def compute_kernel_energy(self, param):
        """Compute the kinetic energy of a kernel"""
        if len(param.shape) == 4:  # Convolutional kernel
            return 0.5 * torch.sum(param ** 2) / (param.shape[0] * param.shape[1])
        return 0.5 * torch.sum(param ** 2)

    def compute_potential(self, loss):
        """Compute potential energy from loss"""
        return loss

    def adjust_layer_scale(self, name, grad):
        """Adjust scaling factor based on gradient behavior"""
        self.grad_history[name].append(grad.norm().item())
        if len(self.grad_history[name]) > self.max_history:
            self.grad_history[name].pop(0)

        if len(self.grad_history[name]) > 1:
            grad_std = np.std(self.grad_history[name])
            grad_mean = np.mean(self.grad_history[name])

            # Adjust scale based on gradient statistics
            if grad_std > grad_mean * 0.5:  # High variance
                self.layer_scales[name] *= 0.95
            else:
                self.layer_scales[name] = min(1.0, self.layer_scales[name] * 1.05)

    def step(self, loss):
        """Perform one optimization step"""
        total_energy = self.compute_potential(loss)

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.grad is None:
                    continue

                # Compute kinetic energy
                kinetic = self.compute_kernel_energy(param)
                total_energy += kinetic

                # Adjust scaling based on layer type and gradient behavior
                self.adjust_layer_scale(name, param.grad)

                # Special handling for convolutional layers
                if len(param.shape) == 4:  # Conv layer
                    # Normalize gradients by kernel size
                    grad_scale = np.prod(param.shape[2:])  # kernel height * width
                    effective_grad = param.grad / grad_scale
                else:  # Fully connected layer
                    effective_grad = param.grad

                # Update velocity with layer-specific scaling
                self.model.velocities[name] = (
                        self.gamma * self.model.velocities[name] +
                        self.alpha * effective_grad * self.layer_scales[name]
                )

                # Apply update
                param.data -= self.model.velocities[name]

        return total_energy

def evaluate_model(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    # Disable gradient calculation for evaluation
    with torch.no_grad():
        for inputs, labels in test_loader:
            # Move inputs and labels to the appropriate device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)

            # Get predictions
            _, predicted = torch.max(outputs.data, 1)

            # Update total and correct counts
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate accuracy
    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test images: {accuracy:.2f}%')
    return accuracy

# Training loop example
def train_cnn_with_lap(model, train_loader, epochs=100):
    criterion = nn.CrossEntropyLoss()
    optimizer = LAPOptimizerCNN(model)

    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Zero gradients
            model.zero_grad()

            # Backward pass
            loss.backward()

            # Optimize with LAP
            total_energy = optimizer.step(loss.item())

            # Print statistics
            running_loss += loss.item()
            if i % 200 == 199:
                print(f'[{epoch + 1}, {i + 1}] '
                      f'loss: {running_loss / 200:.3f} '
                      f'energy: {total_energy:.3f}')
                running_loss = 0.0

def imshow(img):
    img = img / 2 + 0.5  # Unnormalize (assuming normalization was [-1, 1])
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')

# Usage example
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class LAPCNN(nn.Module):
    def __init__(self):
        super(LAPCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)

        # Initialize velocities for LAP
        self.init_velocities()

    def init_velocities(self):
        self.velocities = {}
        for name, param in self.named_parameters():
            self.velocities[name] = torch.zeros_like(param.data)

    def forward(self, x):
        # Convolutional layers with max pooling
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)

        # Flatten and fully connected layers
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x




class LAPOptimizerCNN:
    def __init__(self, model, alpha=0.01, gamma=0.9):
        """
        LAP Optimizer adapted for CNNs
        args:
            model: The neural network model
            alpha: Learning rate
            gamma: Momentum coefficient
        """
        self.model = model
        self.alpha = alpha
        self.gamma = gamma

        # Initialize adaptive parameters
        self.layer_scales = defaultdict(lambda: 1.0)
        self.grad_history = defaultdict(list)
        self.max_history = 50

    def compute_kernel_energy(self, param):
        """Compute the kinetic energy of a kernel"""
        if len(param.shape) == 4:  # Convolutional kernel
            return 0.5 * torch.sum(param ** 2) / (param.shape[0] * param.shape[1])
        return 0.5 * torch.sum(param ** 2)

    def compute_potential(self, loss):
        """Compute potential energy from loss"""
        return loss

    def adjust_layer_scale(self, name, grad):
        """Adjust scaling factor based on gradient behavior"""
        self.grad_history[name].append(grad.norm().item())
        if len(self.grad_history[name]) > self.max_history:
            self.grad_history[name].pop(0)

        if len(self.grad_history[name]) > 1:
            grad_std = np.std(self.grad_history[name])
            grad_mean = np.mean(self.grad_history[name])

            # Adjust scale based on gradient statistics
            if grad_std > grad_mean * 0.5:  # High variance
                self.layer_scales[name] *= 0.95
            else:
                self.layer_scales[name] = min(1.0, self.layer_scales[name] * 1.05)

    def step(self, loss):
        """Perform one optimization step"""
        total_energy = self.compute_potential(loss)

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.grad is None:
                    continue

                # Compute kinetic energy
                kinetic = self.compute_kernel_energy(param)
                total_energy += kinetic

                # Adjust scaling based on layer type and gradient behavior
                self.adjust_layer_scale(name, param.grad)

                # Special handling for convolutional layers
                if len(param.shape) == 4:  # Conv layer
                    # Normalize gradients by kernel size
                    grad_scale = np.prod(param.shape[2:])  # kernel height * width
                    effective_grad = param.grad / grad_scale
                else:  # Fully connected layer
                    effective_grad = param.grad

                # Update velocity with layer-specific scaling
                self.model.velocities[name] = (
                        self.gamma * self.model.velocities[name] +
                        self.alpha * effective_grad * self.layer_scales[name]
                )

                # Apply update
                param.data -= self.model.velocities[name]

        return total_energy



def evaluate_model(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    # Disable gradient calculation for evaluation
    with torch.no_grad():
        for inputs, labels in test_loader:
            # Move inputs and labels to the appropriate device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)

            # Get predictions
            _, predicted = torch.max(outputs.data, 1)

            # Update total and correct counts
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate accuracy
    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test images: {accuracy:.2f}%')
    return accuracy


# Training loop example
def train_cnn_with_lap(model, train_loader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = LAPOptimizerCNN(model)
    loss_values = []

    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            # Move inputs and labels to device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Zero gradients
            model.zero_grad()

            # Backward pass
            loss.backward()

            # Optimize with LAP
            total_energy = optimizer.step(loss.item())

            # Accumulate loss
            running_loss += loss.item()
            loss_values.append(loss.item())

            # Print statistics
            if i % 200 == 199:
                print(f'[LAP] [{epoch + 1}, {i + 1}] '
                      f'loss: {running_loss / 200:.3f} '
                      f'energy: {total_energy:.3f}')
                running_loss = 0.0
    return loss_values


def train_cnn_with_gd(model, train_loader, epochs=10, learning_rate=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0)
    loss_values = []

    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            # Move inputs and labels to device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item()
            loss_values.append(loss.item())

            # Print statistics
            if i % 200 == 199:
                print(f'[GD] [{epoch + 1}, {i + 1}] loss: {running_loss / 200:.3f}')
                running_loss = 0.0
    return loss_values


def imshow(img):
    img = img / 2 + 0.5  # Unnormalize (assuming normalization was [-1, 1])
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')

# Usage example
if __name__ == "__main__":
    # Set random seeds for reproducibility
    import random
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize dataset and dataloader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=2)

    # Load the test dataset
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

    # CIFAR-10 class names
    classes = ['plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

    # Initialize models
    model_lap = LAPCNN().to(device)
    model_gd = LAPCNN().to(device)

    # Train the model with LAP optimizer
    print("\nTraining with LAP Optimizer:")
    lap_loss_values = train_cnn_with_lap(model_lap, trainloader, epochs=10)

    # Evaluate the model trained with LAP
    print("\nEvaluating model trained with LAP:")
    test_accuracy_lap = evaluate_model(model_lap, test_loader)
    print(f"\nTest Accuracy with LAP Optimizer: {test_accuracy_lap:.2f}%")

    # Train the model with Gradient Descent
    print("\nTraining with Gradient Descent:")
    gd_loss_values = train_cnn_with_gd(model_gd, trainloader, epochs=10, learning_rate=0.01)

    # Evaluate the model trained with GD
    print("\nEvaluating model trained with Gradient Descent:")
    test_accuracy_gd = evaluate_model(model_gd, test_loader)
    print(f"\nTest Accuracy with Gradient Descent: {test_accuracy_gd:.2f}%")

    # Plot training loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(np.convolve(lap_loss_values, np.ones(100)/100, mode='valid'), label='LAP Optimizer')
    plt.plot(np.convolve(gd_loss_values, np.ones(100)/100, mode='valid'), label='Gradient Descent')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison (Smoothed)')
    plt.legend()
    plt.show()

    # Display sample predictions for LAP model
    print("\nSample predictions from model trained with LAP:")
    dataiter = iter(test_loader)
    images, labels = next(dataiter)

    images = images.to(device)
    outputs = model_lap(images)
    _, predicted = torch.max(outputs, 1)

    images = images.cpu()
    predicted = predicted.cpu()
    labels = labels.cpu()

    fig = plt.figure(figsize=(12, 4))
    for idx in np.arange(8):
        ax = fig.add_subplot(2, 4, idx + 1, xticks=[], yticks=[])
        imshow(images[idx])
        ax.set_title(f"Pred: {classes[predicted[idx]]}\nTrue: {classes[labels[idx]]}")

    plt.tight_layout()
    plt.show()

    # Display sample predictions for GD model
    print("\nSample predictions from model trained with GD:")
    dataiter = iter(test_loader)
    images, labels = next(dataiter)

    images = images.to(device)
    outputs = model_gd(images)
    _, predicted = torch.max(outputs, 1)

    images = images.cpu()
    predicted = predicted.cpu()
    labels = labels.cpu()

    fig = plt.figure(figsize=(12, 4))
    for idx in np.arange(8):
        ax = fig.add_subplot(2, 4, idx + 1, xticks=[], yticks=[])
        imshow(images[idx])
        ax.set_title(f"Pred: {classes[predicted[idx]]}\nTrue: {classes[labels[idx]]}")

    plt.tight_layout()
    plt.show()