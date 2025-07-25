import time

import numpy as np
import torch
from main import MyNeuronSystem
from torch.utils.data import DataLoader, TensorDataset
from ml.generate_.geometrical.generate_geometrical import generate_circle_points, generate_spiral_points
import os
from pathlib import Path

# def generate_data():
#     X = torch.randn(100, 8)  # 100 próbek, 8 cech
#     y = torch.randint(0, 3, (100,))  # 3 klasy
#     return X, y

def generate_data():
    circle = generate_circle_points(n_points=100)
    spiral = generate_spiral_points(n_points=100)

    X = np.vstack([circle, spiral])  # shape: (200, 2)
    y = np.array([0] * 100 + [1] * 100)  # 0 = circle, 1 = spiral

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    return X_tensor, y_tensor


def timestamp():
    return time.strftime("%Y%m%d-%H%M%S")


def save_model_stamp(model):
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    fname = checkpoint_dir / f"model-{timestamp()}.pt"
    torch.save(model.state_dict(), fname)
    print(f"[✓] Model saved to {fname}")


def train():
    model = MyNeuronSystem(2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    X, y = generate_data()
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=10, shuffle=True)

    for epoch in range(20):
        total_loss = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs, _ = model(batch_X)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(loader):.4f}")

    save_model_stamp(model)


if __name__ == "__main__":
    train()
