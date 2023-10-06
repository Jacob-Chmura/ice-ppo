import torch.nn as nn
import torch.nn.functional as F

class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super().__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super().__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList(
            [Residual(in_channels, num_hiddens, num_residual_hiddens) for _ in range(self._num_residual_layers)]
        )

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens):
        super().__init__()
        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens//4,
                                 kernel_size=3,
                                 stride=2, padding=1)
        self._mp_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens//4,
                                 out_channels=num_hiddens//2,
                                 kernel_size=3,
                                 stride=2, padding=1)
        self._mp_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens//2,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=2, padding=1)
        self._mp_3 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, inputs):
        x = F.relu(self._conv_1(inputs))
        x = self._mp_1(x)
        x = F.relu(self._conv_2(x))
        x = self._mp_2(x)
        x = self._conv_3(x)
        x = self._mp_3(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens):
        super().__init__()
        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=in_channels,
                                                out_channels=in_channels//2,
                                                kernel_size=6,
                                                stride=3)
        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=in_channels//2,
                                                out_channels=num_hiddens,
                                                kernel_size=6,
                                                stride=3)
        self._conv_trans_3 = nn.ConvTranspose2d(in_channels=num_hiddens,
                                                out_channels=3,
                                                kernel_size=6,
                                                stride=3, padding=1)

    def forward(self, inputs):
        inputs = inputs.unsqueeze(-1).unsqueeze(-1)
        x = F.relu(self._conv_trans_1(inputs))
        x = F.relu(self._conv_trans_2(x))
        x = self._conv_trans_3(x)
        return x
