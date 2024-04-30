import os.path

import matplotlib.pyplot as plt

from util.differntial_entropy.network import Network, ContraNetwork
import numpy as np
import torch
from util.dataloader import get_train_data

NORMALFACTOR = (0.5 * np.log(2 * np.pi * np.e))

# parameters to measure in
VARS = np.arange(10.1, 15.2, 1.0)
VARS = [round(var, 2) for var in VARS]
print(VARS)
DEPTH = 10 # 50  # 25

# load training data
train_data, test_data = get_train_data(bs=100)

# determine device to perform calculations on
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Device: {device} with {torch.cuda.device_count()} gpus")

# create folder to store data in
PATH = f"data/nice_images_deeper_70"
if os.path.exists(PATH):
    pass
    # raise AttributeError("Path exists!")
else:
    os.mkdir(PATH)

# for each variance perform measurement
for var in VARS:
    print(f"\rStart {var}")

    # create network
    net = Network(28 ** 2, 10, DEPTH, (var, 0.05), device=device)  # input should be 28**2 for MNIST data
    conet = ContraNetwork(net, device=device)
    td_lvl = conet.get_activations(train_data)  # [[inputactivations1], [inputactivations2], ...]
    td_lvl = tuple(zip(*td_lvl))  #[[activations1], [activations2], ...]

    for lvl in range(len(conet)):
        print(f"{lvl} done")
        # output: td_lvl[lvl]
        # input: td_lvl[lvl + 1]
        # wanted: [[input1, output1], [input2, output2], ...]
        td = zip(td_lvl[lvl + 1], td_lvl[lvl])
        conet.train_lvl(lvl, td)
    exit()

    # train reconstruction networks
    conet.train(train_data)
    print("Conet trained")

    for c, (inp, _) in zip(range(1), test_data):
        entropies = []
        cascades = conet.cascade(inp)

        for cascade in cascades:
            # calculate the differential entropy
            cascade /= torch.sum(cascade, dim=1)[:, None]
            # cascade [batch, data] -> normalize over batch
            std = torch.std(cascade, dim=0)
            entropy = torch.mean(NORMALFACTOR + torch.log(std))
            entropies.append(entropy)

            #fig, axes = plt.subplots(3, 3)
            #axes = axes.flatten()
            #for i in range(8):
            #    axes[i].imshow(cascade[i].numpy().reshape(28, 28), aspect="auto")
            #axes[8].imshow(std.numpy().reshape(28, 28), aspect="auto")
            #plt.show()

        _, accs = net.training(train_data, test_data, 0)  # 10
        print("Trained")
        plt.plot(entropies)
        plt.grid()
        if len(accs) == 0:
            plt.title("Not tested")
        elif accs[-1] > 0.12:
            plt.title("learns")
        else:
            plt.title("Does not learn")
        plt.xlabel("Depth of the network")
        plt.ylabel("Differential entropy")
        plt.savefig(f"{PATH}/{var}.png")
        plt.close()
