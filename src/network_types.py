import torch
import torch.nn as nn
from network_modified import ResBlock, Channel, ContraNetwork, View, Cut, CoTanh, Hist

PRINTVIEW = False


def get_conv_net(depth, c=64):
    layers = [
        torch.nn.Sequential(nn.Conv2d(1, c, kernel_size=(3, 3), stride=1, padding="same"), nn.Tanh()),
        torch.nn.Sequential(nn.Conv2d(c, c, kernel_size=(3, 3), stride=2, padding=1), nn.Tanh()),
        torch.nn.Sequential(nn.Conv2d(c, c, kernel_size=(3, 3), stride=2, padding=1), nn.Tanh()),
    ]
    layer_conet = [
        nn.Sequential(nn.ConvTranspose2d(c, 1, kernel_size=(3, 3)), nn.Tanh(), Cut()),
        nn.Sequential(nn.ConvTranspose2d(c, c, kernel_size=(3, 3), stride=2), nn.Tanh(), Cut("end")),
        nn.Sequential(nn.ConvTranspose2d(c, c, kernel_size=(3, 3), stride=2), nn.Tanh(), Cut("end")),
    ]

    for i in range(depth - 1):
        layers.append(
            torch.nn.Sequential(nn.Conv2d(c, c, kernel_size=(3, 3)), nn.Tanh(), nn.CircularPad2d(1))
        )

        layer_conet.append(
            nn.Sequential(
                nn.Sequential(nn.ConvTranspose2d(c, c, kernel_size=(3, 3)), nn.Tanh(), Cut())
            ),
        )

    # Flatten at the end
    layers.append(
        nn.Sequential(nn.Conv2d(c, c, kernel_size=(3, 3)), nn.Tanh(), nn.AvgPool2d((5, 5)), nn.Flatten()))

    layer_sizes = [c, 10]
    for l1, l2 in zip(layer_sizes, layer_sizes[1:]):
        layers.append(
            nn.Sequential(
                torch.nn.Linear(l1, l2, bias=False),
                nn.Tanh()
            )
        )
    return layers, layer_conet


def get_red_resnet(depth, size):
    layers = []
    layers_conet = []

    layer_sizes = [size] * depth

    # resnet
    for l1, l2 in zip(layer_sizes, layer_sizes[1:]):
        # resnet
        layers.append(
            ResBlock(
                nn.Sequential(nn.Linear(l1, l2), nn.Tanh()),
                nn.Sequential(nn.Identity())
            )
        )

        # conet
        layers_conet.append(
            nn.Sequential(
                torch.nn.Linear(l2, l1),
                nn.Tanhshrink(),
                torch.nn.Linear(l1, l1),
                nn.Tanhshrink()
            )
            #torch.nn.Linear(l2, l1),
            #nn.ReLU(),
            #torch.nn.Linear(l2, l1),
            # nn.Tanhshrink()
            #CoTanh
        )

    layer_sizes = [size] + [400, 10]
    for l1, l2 in zip(layer_sizes, layer_sizes[1:]):
        # resnet
        layers.append(
            nn.Sequential(nn.Linear(l1, l2), nn.Tanh()),
        )

        # conet
        layers_conet.append(
            nn.Sequential(
                torch.nn.Linear(l2, l1),
                nn.Tanh()
            )
        )

    return layers, layers_conet


def get_resnet_normal(depth, c = 12):
    layers = [
        torch.nn.Sequential(nn.Conv2d(1, c, kernel_size=(3, 3), stride=1, padding="same"), nn.Tanh()),
        torch.nn.Sequential(nn.Conv2d(c, c, kernel_size=(3, 3), stride=2, padding=1), nn.Tanh()),
        torch.nn.Sequential(nn.Conv2d(c, c, kernel_size=(3, 3), stride=2, padding=1), nn.Tanh()),
    ]
    layer_conet = [
        nn.Sequential(nn.ConvTranspose2d(c, 1, kernel_size=(3, 3)), nn.Tanh(), Cut()),
        nn.Sequential(nn.ConvTranspose2d(c, c, kernel_size=(3, 3), stride=2), nn.Tanh(), Cut("end")),
        nn.Sequential(nn.ConvTranspose2d(c, c, kernel_size=(3, 3), stride=2), nn.Tanh(), Cut("end")),
    ]

    for i in range(depth - 1):
        layers.append(
            ResBlock(
                torch.nn.Sequential(nn.Conv2d(c, c, kernel_size=(3, 3)), nn.CircularPad2d(1), nn.Tanh(),
                                    nn.Conv2d(c, c, kernel_size=(3, 3)), nn.CircularPad2d(1)),
                torch.nn.Sequential(nn.Identity())
            )
            )

        layer_conet.append(
            nn.Sequential(
                nn.ConvTranspose2d(c, c, kernel_size=(3, 3)), nn.Tanh(), Cut(),
                nn.ConvTranspose2d(c, c, kernel_size=(3, 3)), nn.Tanh(), Cut()
            )
        )

    # Flatten at the end
    layers.append(
        nn.Sequential(nn.Conv2d(c, c, kernel_size=(3, 3)), nn.Tanh(), nn.AvgPool2d((5, 5)), nn.Flatten()))

    layer_sizes = [c, 10]
    for l1, l2 in zip(layer_sizes, layer_sizes[1:]):
        layers.append(
            nn.Sequential(
                torch.nn.Linear(l1, l2, bias=False),
                nn.Tanh()
            )
        )
    return layers, layer_conet