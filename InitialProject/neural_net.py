import torch
from torch import nn
from torch.nn import functional as F


class FullyConnectedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, p_dropout=0.1, skip_connection=False, activation=nn.LeakyReLU):
        super().__init__()
        self.layer = nn.Linear(in_channels, out_channels)
        self.activation = activation()
        self.dropout = nn.Dropout(p_dropout)
        self.skip_connection = skip_connection

        assert (not skip_connection) or (in_channels == out_channels)

    def forward(self, x):
        out = self.layer(x)
        out = self.dropout(out)
        out = self.activation(out)
        if self.skip_connection:
            out += x
        return out


class FullyConnectedModel(nn.Module):
    def __init__(self, in_channels, out_channels, decode_channels, hidden_channels, n_layers,
                 p_dropout=0.1, activation=nn.LeakyReLU, final_activation=None):
        super().__init__()
        layers = [nn.Linear(in_channels, hidden_channels), activation()]

        for _ in range(n_layers):
            layers.append(FullyConnectedBlock(hidden_channels, hidden_channels, p_dropout,
                                              skip_connection=True, activation=activation))
        self.encode = nn.Sequential(*layers)
        if final_activation is None:
            self.decode = nn.Sequential(
                nn.Linear(hidden_channels, decode_channels),
                activation(),
                nn.Linear(decode_channels, out_channels)
            )
        else:
            self.decode = nn.Sequential(
                nn.Linear(hidden_channels, decode_channels),
                activation(),
                nn.Linear(decode_channels, out_channels),
                final_activation
            )

    def forward(self, x):
        return self.decode(self.encode(x))


def main():
    model = FullyConnectedModel(160, 1, 40, 80, 2)


if __name__ == "__main__":
    main()
