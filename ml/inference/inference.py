import torch

from ml.utils.load_model import load_model


def classify(model, x):
    with torch.no_grad():
        out1, out2 = model(x)
        pred_class = torch.argmax(out1, dim=1)
    return pred_class

if __name__ == "__main__":
    model_path = "../checkpoints/model-20250725-155025.pt"
    model = load_model(model_path, input_dim=2)
    x_new = torch.tensor([[0.1, 0.2]], dtype=torch.float32)
    prediction = classify(model, x_new)
    print(f"Predykcja klasy dla {x_new.numpy()}: {prediction.item()}")
