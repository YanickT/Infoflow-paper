from typing import Tuple
import torch
import torchvision
from torch.utils.data import DataLoader


DATAPATH = ""


def get_train_data(bs: int = 64) -> Tuple[DataLoader, DataLoader]:
    """
    Loads training data form default directory given by global variable PATH.
    :param bs: int = batchsize
    :return: Tuple[torch.Dataloader, torch.Dataloader] = Dataloader for training and testing
    """
    trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Lambda(lambda x: torch.flatten(x))])
    train_data = torchvision.datasets.MNIST(DATAPATH, train=True, download=True, transform=trans)
    test_data = torchvision.datasets.MNIST(DATAPATH, train=False, transform=trans, download=True)
    return DataLoader(train_data, batch_size=bs, shuffle=True), DataLoader(test_data, batch_size=bs, shuffle=True)


def get_train_data_cifar(bs: int = 64) -> Tuple[DataLoader, DataLoader]:
    """
    Loads training data form default directory given by global variable PATH.
    :param bs: int = batchsize
    :return: Tuple[torch.Dataloader, torch.Dataloader] = Dataloader for training and testing
    """
    DATAPATH = "CIFAR10"
    trans = torchvision.transforms.Compose([torchvision.transforms.Grayscale(num_output_channels=1),
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Lambda(lambda x: torch.flatten(x))])
    train_data = torchvision.datasets.CIFAR10(DATAPATH, train=True, download=True, transform=trans)
    test_data = torchvision.datasets.CIFAR10(DATAPATH, train=False, transform=trans, download=True)
    return DataLoader(train_data, batch_size=bs, shuffle=True), DataLoader(test_data, batch_size=bs, shuffle=True)


if __name__ == "__main__":
    # plot a image of the first example in the data
    import matplotlib.pyplot as plt

    data, test = get_train_data()
    for test, target in data:
        plt.tight_layout()
        plt.imshow(test[0].reshape(28, 28), cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(target[0]))
        plt.xticks([])
        plt.yticks([])
        plt.show()
        break
