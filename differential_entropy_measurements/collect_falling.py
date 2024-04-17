import os.path

import similarity_based_cutoff as sbc
import numpy as np
import torch
from util.dataloader import get_train_data, get_train_data_cifar
import matplotlib.pyplot as plt
torch.backends.cudnn.benchmark = True


def evaluate(depth, var):
    # train middle one
    net = sbc.Network(28 ** 2, 10, depth, [var, 0.05], device=device)
    conet = sbc.ContraNetwork(net, device=device)

    # evaluate trainability
    conet.train(train_data)
    cutoffs = []
    for threshold in CUTOFFS:
        temp = []
        for c, (inp, _) in zip(range(10), test_data):
            cutoff = conet.cutoff_cascade(inp, threshold)
            temp.append(cutoff)
            # cutoffs.append(cutoff)

        if any([(e is None) for e in temp]):
            cutoff = None
        else:
            cutoff = int(round(np.mean(temp)))
        cutoffs.append(cutoff)

    with open(f"{PATH}/conet_{var}_{depth}.txt", "w") as doc:
        doc.writelines("\n".join([f"{t};{c}" for t, c in zip(CUTOFFS, cutoffs)]))

    # train network to show trainability (in multiple steps)

    accs = []
    for i in range(EPOCHS):
        print(f"\rTraining {i} / {EPOCHS}", end="")
        loss, accs_ = net.training(train_data, test_data, 1)
        accs.append(accs_[0])

    with open(f"{PATH}/net_{var}_{depth}.txt", "w") as doc:
        doc.writelines("\n".join([f"{i};{a}" for i, a in enumerate(accs)]))

    return accs


STARTVAR = 1.6
VARRANGE = [0.1, 3.1]
VARSTEP = 0.1

STARTDEPTH = 40
DEPTHSTEP = 5
EPOCHS = 20
CUTOFFS = np.linspace(-1.0, -5.5, 10)

PATH = f"dummy/2ndMethod/{STARTDEPTH}_wide_{EPOCHS}eps_entropy_falling_measurement"
if os.path.exists(PATH):
    raise AttributeError("Path exists!")
else:
    os.mkdir(PATH)


train_data, test_data = get_train_data()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Device: {device} with {torch.cuda.device_count()} gpus")


left = STARTDEPTH
right = STARTDEPTH
left_var = STARTVAR - VARSTEP
right_var = STARTVAR + VARSTEP

evaluate(STARTDEPTH, STARTVAR)
while True:
    print(f"Train: {left_var} | {right_var}")

    while True:
        acc_left = evaluate(left, left_var)[-1]
        if acc_left > 0.2:
            break
        left -= DEPTHSTEP
        print(f"Update left depth to: {acc_left}")
    left_var -= VARSTEP

    while True:
        acc_right = evaluate(right, right_var)[-1]
        if acc_right > 0.2:
            break
        right -= DEPTHSTEP
        print(f"Update right depth to: {acc_right}")
    right_var -= VARSTEP

    if right_var > VARRANGE[1] or left_var < VARRANGE[0]:
        break
