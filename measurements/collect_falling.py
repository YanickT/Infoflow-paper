from typing import List, Union
from src.network import Network
from src.dataloader import get_train_data, get_train_data_cifar
import os
import torch
torch.backends.cudnn.benchmark = True


# parameter for data collection
STARTVAR = 1.7
VARRANGE = [0.1, 4.0]
DIRECTION = -1  # -1

ACCTHRESHOLD = 0.4
VARSTEP = 0.1
STARTDEPTH = 150
DEPTHSTEP = 5
EPOCHS = 200
PATH = f"data/collect_falling"


# function for evaluation of network configuration
def evaluate(depth: int, var_: float, train_data, test_data, device: Union[torch.device, str] = "cpu") -> List[float]:
    """
    Train a network for a given variance var for the weights
    :param depth: int = number of layers of the network
    :param var: float = weight-variance for initialization
    :param device: Union[torch.device, str] = device to perform calculation on
    :return: List[float] = accuracy(epoch)
    """
    net = Network(32 ** 2, 10, depth, (var_, 0.05), device=device, size="shrinking")

    accs = []
    for i in range(EPOCHS):
        loss, accs_ = net.training(train_data, test_data, 1, verbose=False)
        accs.append(accs_[0])
        print(f"\rTraining {i + 1} / {EPOCHS} : {accs_[0]}", end="")
        if accs_[0] > ACCTHRESHOLD:
            break
    print("")
    with open(f"{PATH}/net_{var_}_{depth}.txt", "w") as doc:
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
train_data, test_data = get_train_data_cifar(bs=64)

# configuration parameter

depth = STARTDEPTH
var_ = STARTVAR

while True:
    print(f"Train: {var_}")

    while True:
        acc = evaluate(depth, var_, train_data, test_data, device)[-1]
        if acc > ACCTHRESHOLD:
            break
        depth -= DEPTHSTEP
        print(f"Update left depth to: {depth}")
    var_ += DIRECTION * VARSTEP
    var_ = round(var_, 2)

    if not (VARRANGE[0] <= var_ <= VARRANGE[1]):
        break
