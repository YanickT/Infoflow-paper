from typing import List, Union
from src.network import Network
from src.dataloader import get_train_data
import os
import torch

# parameter for data collection
STARTVAR = 1.6
VARRANGE = [0.1, 5.1]
DIRECTION = -1  # -1

ACCTHRESHOLD = 0.8
VARSTEP = 0.25
STARTDEPTH = 50
DEPTHSTEP = 10
EPOCHS = 1
PATH = f"data/{STARTDEPTH}_{EPOCHS}_falling_MNIST"


# function for evaluation of network configuration
def evaluate(depth: int, var: float, device: Union[torch.device, str] = "cpu") -> List[float]:
    """
    Train a network for a given variance var for the weights
    :param depth: int = number of layers of the network
    :param var: float = weight-variance for initialization
    :param device: Union[torch.device, str] = device to perform calculation on
    :return: List[float] = accuracy(epoch)
    """
    net = Network(28 ** 2, 10, depth, (var, 0.05), device=device)

    accs = []
    for i in range(EPOCHS):
        print(f"\rTraining {i} / {EPOCHS}", end="")
        loss, accs_ = net.training(train_data, test_data, 1)
        accs.append(accs_[0])

    with open(f"{PATH}/net_{var}_{depth}.txt", "w") as doc:
        doc.writelines("\n".join([f"{i};{a}" for i, a in enumerate(accs)]))

    return accs


# create path to store information in
if os.path.exists(PATH):
    pass
    # raise AttributeError("Path exists!")
else:
    os.mkdir(PATH)

# load device to perform calculations on
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Device: {device} with {torch.cuda.device_count()} gpus")

# get training data
train_data, test_data = get_train_data()

# configuration parameter
depth = STARTDEPTH
var_ = STARTVAR + DIRECTION * VARSTEP

evaluate(STARTDEPTH, STARTVAR)
while True:
    print(f"Train: {var_}")

    while True:
        acc = evaluate(depth, var_, device)[-1]
        if acc > ACCTHRESHOLD:
            break
        depth -= DEPTHSTEP
        print(f"Update left depth to: {depth}")
    var_ -= VARSTEP

    if not (VARRANGE[0] <= var_ <= VARRANGE[1]):
        break
