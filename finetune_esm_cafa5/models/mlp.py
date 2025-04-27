from abc import abstractmethod
from typing import List, Type

import torch
from dataclasses import dataclass, field
from torch import nn

@dataclass
class ClassifierConfig:
    input_size: int
    output_size: int

    @abstractmethod
    def get_model(self) -> nn.Module:
        pass

@dataclass
class MLPConfig(ClassifierConfig):
    dropout: float = 0.1
    hidden_sizes: List[int] = field(default_factory=lambda: [])

    def get_model(self):
        return MLP(self)

class MLP(nn.Module):
    def __init__(self, config: MLPConfig):
        super(MLP, self).__init__()

        self.config = config

        layer_sizes = [config.input_size, *config.hidden_sizes, config.output_size]

        self.layers = nn.ModuleList([
            nn.Linear(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes)-1)
        ])
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.act(self.layers[0](x))
        for layer in self.layers[1:-1]:
            x = self.act(layer(self.dropout(x)))
        x = self.layers[-1](x)
        return x


if __name__ == '__main__':
    mlp_config = MLPConfig(input_size=3, output_size=1, dropout=0.1)

    print(mlp_config.__dict__)
