import os.path
from util.differntial_entropy.network import Network, ContraNetwork
import numpy as np
import torch
from util.dataloader import get_train_data

# parameters to measure in
VARS = np.arange(0.1, 4.0, 0.5)
VARS = [round(var, 2) for var in VARS]
print(VARS)
DEPTH = 10
EPOCHS = 10
CUTOFFS = np.linspace(-1.0, -3.5, 6)
print(CUTOFFS)

# load training data
train_data, test_data = get_train_data()

# determine device to perform calculations on
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Device: {device} with {torch.cuda.device_count()} gpus")

# create folder to store data in
PATH = f"data/{DEPTH}_wide_{EPOCHS}eps_MNIST"
if os.path.exists(PATH):
    raise AttributeError("Path exists!")
else:
    os.mkdir(PATH)

# for each variance perform measurement
for var in VARS:
    print(f"\rStart {var}")

    # create network
    net = Network(28 ** 2, 10, DEPTH, (var, 0.05), device=device)  # input should be 28**2 for MNIST data
    conet = ContraNetwork(net, device=device)

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

    with open(f"{PATH}/conet_{var}.txt", "w") as doc:
        doc.writelines("\n".join([f"{t};{c}" for t, c in zip(CUTOFFS, cutoffs)]))

    # train network to show trainability (in multiple steps)
    accs = []
    for i in range(EPOCHS):
        print(f"\rTraining {i} / {EPOCHS}", end="")
        loss, accs_ = net.training(train_data, test_data, 1)
        accs.append(accs_[0])

    with open(f"{PATH}/net_{var}.txt", "w") as doc:
        doc.writelines("\n".join([f"{i};{a}" for i, a in enumerate(accs)]))
