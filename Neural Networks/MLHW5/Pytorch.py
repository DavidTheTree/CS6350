import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Load Data Function
def load_data():
    train_data = pd.read_csv('./BankNote/train.csv', header=None)
    test_data = pd.read_csv('./BankNote/test.csv', header=None)

    attributes = ['variance', 'skewness', 'curtosis', 'entropy', 'genuineorforged']
    train_data.columns = attributes
    test_data.columns = attributes

    return train_data, test_data, attributes

# Neural Network Model
def create_model(input_dim, output_dim, hidden_dim, depth, activation):
    layers = []

    # Input Layer
    layers.append(nn.Linear(input_dim, hidden_dim))
    layers.append(activation)

    # Hidden Layers
    for _ in range(depth - 2):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(activation)

    # Output Layer
    layers.append(nn.Linear(hidden_dim, output_dim))
    layers.append(nn.Sigmoid())  # For binary classification

    return nn.Sequential(*layers)

# Initialization Function
def initialize_weights(model, activation):
    for layer in model:
        if isinstance(layer, nn.Linear):
            if activation == nn.Tanh():
                nn.init.xavier_uniform_(layer.weight)
            elif activation == nn.ReLU():
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            nn.init.zeros_(layer.bias)

# Training Function
def train_model(model, train_loader, test_loader, criterion, optimizer, epochs):
    train_errors, test_errors = [], []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

        train_errors.append(1 - correct / total)

        # Evaluate on test data
        model.eval()
        with torch.no_grad():
            test_correct = 0
            test_total = 0
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch)
                predicted = (outputs > 0.5).float()
                test_correct += (predicted == y_batch).sum().item()
                test_total += y_batch.size(0)

            test_errors.append(1 - test_correct / test_total)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Train Loss: {total_loss:.4f}, Train Error: {train_errors[-1]:.4f}, Test Error: {test_errors[-1]:.4f}")

    return train_errors, test_errors

# Main Execution
if __name__ == "__main__":
    # Load data
    train_data, test_data, attributes = load_data()
    X_train = train_data[attributes[:-1]].values
    y_train = train_data['genuineorforged'].values.reshape(-1, 1)
    X_test = test_data[attributes[:-1]].values
    y_test = test_data['genuineorforged'].values.reshape(-1, 1)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # Data loaders
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Parameters
    depths = [3, 5, 9]
    widths = [5, 10, 25, 50, 100]
    epochs = 100
    activation_functions = {'tanh': nn.Tanh(), 'relu': nn.ReLU()}

    results = {}

    for activation_name, activation_fn in activation_functions.items():
        for depth in depths:
            for width in widths:
                print(f"Training with activation: {activation_name}, depth: {depth}, width: {width}")

                model = create_model(X_train.shape[1], 1, width, depth, activation_fn)
                initialize_weights(model, activation_fn)

                criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
                optimizer = optim.Adam(model.parameters(), lr=1e-3)

                train_errors, test_errors = train_model(model, train_loader, test_loader, criterion, optimizer, epochs)

                results[(activation_name, depth, width)] = {
                    'train_errors': train_errors,
                    'test_errors': test_errors
                }

    # Report results
    for key, value in results.items():
        activation, depth, width = key
        print(f"Activation: {activation}, Depth: {depth}, Width: {width}")
        print(f"  Final Train Error: {value['train_errors'][-1]:.4f}")
        print(f"  Final Test Error: {value['test_errors'][-1]:.4f}")
