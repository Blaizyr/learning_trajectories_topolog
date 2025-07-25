import torch

from ml.my_neuron_system import MyNeuronSystem


def load_model(path, input_dim=2):
    model = MyNeuronSystem(input_dim)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model
