import torch
from ml.my_neuron_system import MyNeuronSystem

def main():
    i = 5

    while i <= 10:
        model = MyNeuronSystem(i)
        x = torch.randn(1, i)
        output = model(x)
        print(f"Input dim: {i}, Output: {output}")
        i += 1

if __name__ == "__main__":
    main()
