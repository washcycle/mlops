# source https://www.youtube.com/watch?v=P6NwZVl8ttc&t=1s
# Hyperparameters control the learning process of the model. They are set before the learning process begins.

import torch
import optuna
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

def train_mnist(model, optimizer):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(1):  # Train for 1 epoch for simplicity
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs.view(-1, 28*28))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:  # Print every 100 mini-batches
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
                running_loss = 0.0

    correct = 0
    total = 0
    with torch.no_grad():
        for data in trainloader:
            images, labels = data
            outputs = model(images.view(-1, 28*28))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy

def objective(trial: optuna.Trial):
    n_layers = trial.suggest_int("n_layers", 1, 3)
    layers = []
    in_feat = 28 * 28
    for i in range(n_layers):          
        out_features = trial.suggest_int(f"n_units_l{i}", 4, 128)
        layers.append(nn.Linear(in_feat, out_features))
        layers.append(nn.ReLU())
        in_feat = out_features
    layers.append(nn.Linear(in_feat, 10))
    layers.append(nn.LogSoftmax(dim=1))
    model = nn.Sequential(*layers)

    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    accuracy = train_mnist(model, optimizer)

    return accuracy

study = optuna.create_study(direction="maximize")

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)



