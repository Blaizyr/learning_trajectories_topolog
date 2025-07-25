import torch
import matplotlib.pyplot as plt

from ml.utils.load_model import load_model
from ml.generate_.geometrical.generate_geometrical import generate_circle_points, generate_spiral_points

def classify_points(model, points_tensor):
    with torch.no_grad():
        out1, _ = model(points_tensor)
        preds = torch.argmax(out1, dim=1)
    return preds


def plot_classification(points, predictions, title):
    plt.scatter(points[:, 0], points[:, 1], c=predictions.numpy(), cmap='coolwarm', s=30)
    plt.title(title)
    plt.axis('equal')


def main():
    model_path = "../checkpoints/model-20250725-155025.pt"

    model = load_model(model_path, input_dim=2)

    circle = generate_circle_points(n_points=100)
    spiral = generate_spiral_points(n_points=100)

    circle_tensor = torch.tensor(circle, dtype=torch.float32)
    spiral_tensor = torch.tensor(spiral, dtype=torch.float32)

    pred_circle = classify_points(model, circle_tensor)
    pred_spiral = classify_points(model, spiral_tensor)

    print("Circle predicted classes distribution:", torch.bincount(pred_circle))
    print("Spiral predicted classes distribution:", torch.bincount(pred_spiral))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plot_classification(circle, pred_circle, "Circle classification")

    plt.subplot(1, 2, 2)
    plot_classification(spiral, pred_spiral, "Spiral classification")

    plt.show()


if __name__ == "__main__":
    main()
