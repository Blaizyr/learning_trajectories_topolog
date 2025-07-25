from torch import nn
import torch.nn.functional as F

class MyNeuronSystem(nn.Module):
    def __init__(self, input_dim, output1_dim = 3, output2_dim = 1):
        super().__init__()

        self.input_layer = nn.Linear(input_dim, 32)
        self.hidden_a = nn.Linear(32, 16)
        self.hidden_b = nn.Linear(32, 16)
        self.loop = nn.Linear(16, 16)
        self.output1 = nn.Linear(16, output1_dim)
        self.output2 = nn.Linear(16, output2_dim)

    def forward(self, x):
        x = F.relu(self.input_layer(x))

        # Rozgałęzienie
        a = F.relu(self.hidden_a(x))
        b = F.relu(self.hidden_b(x))

        # Sprzężenie zwrotne
        for _ in range(3):
            a = a + F.relu(self.loop(a))

        # Złączenie
        combined = a + b

        # Wyjścia
        out1 = self.output1(combined)
        out2 = self.output2(combined)
        return out1, out2
