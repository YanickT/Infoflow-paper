from src.network import Network, ContraNetwork
import numpy as np
import torch
import os
from src.dataloader import get_train_data, get_train_data_cifar
from src.entropies import rel_entropy, cutoff_det, diff_entropy
import matplotlib.pyplot as plt

# parameters for the network
DEPTH = 20
VARS = np.arange(1.0, 4.0, 1.0)[::-1]
print(len(VARS))
PATH = f"data/collect_data"

os.mkdir(PATH)
os.mkdir(f"{PATH}/entropy")
os.mkdir(f"{PATH}/data")
os.mkdir(f"{PATH}/images")

# load training data
_, test_data = get_train_data(bs=100)
train_data, _ = get_train_data()


# check for device to use
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Device: {device} with {torch.cuda.device_count()} gpus")


for var_w in VARS:
    # initialize neural network
    net = Network(28 ** 2, 10, DEPTH, (var_w, 0.05), device=device, size="wide")  # input should be 32**2 for CIFAR10 data
    conet = ContraNetwork(net, device=device)

    # train reconstruction
    conet.train(train_data, its=1)

    # show some reconstructions and determined cutoffs
    for inp, _ in test_data:
        cascades = conet.cascade(inp)
        reference = cascades[0]

        # get cutoff from cascades with relative entropy
        entropies = [rel_entropy(reference, sec).tolist() for sec in cascades[1:-2]]
        entropies = np.array(tuple(zip(*entropies)))

        plt.title(f"{var_w}")
        plt.plot(np.average(entropies.transpose(), axis=1))
        plt.savefig(f"{PATH}/images/{var_w:.2f}.png")
        plt.close()

        # determine cutoff of average
        entropies_avg = entropies / entropies[:, -1][:, None]
        avg_entropies = np.median(entropies_avg.transpose(), axis=1).reshape([1, len(entropies[0])])
        rel_avg_cutoff = cutoff_det(avg_entropies, rtol=0.005)[0]
        print(f"Rel. Avg entropy - Cutoff: {rel_avg_cutoff}")

        # determine cutoff of single relative entropies
        cutoffs = cutoff_det(entropies, rtol=0.005).tolist()
        rel_cutoff = (np.mean(cutoffs), np.std(cutoffs, ddof=1))
        print(f"Rel. entropy - Cutoff: {rel_cutoff[0]} $ \\pm $ {rel_cutoff[1]}")

        # get cutoff from cascades with differential entropy
        entropies = [[diff_entropy(sec).tolist() for sec in cascades[1:-2]]]
        with open(f"{PATH}/entropy/diff_entropy_{var_w : .3f}.csv", "w") as doc:
            doc.writelines("\n".join([";".join([str(v) for v in e]) for e in entropies]))
        # plt.plot(entropies[0])
        # plt.show()
        # cutoff = cutoff_det(entropies, atol=5e-1, method="differential")[0]
        cutoff = cutoff_det(entropies, rtol=0.005)[0]
        print(f"Dif. entropy - Cutoff: {cutoff}")

        with open(f"{PATH}/data/entropy_{var_w : .3f}.csv", "w") as doc:
            doc.writelines([f"Rel. entropy; {rel_cutoff[0]}; {rel_cutoff[1]}\n",
                            f"Dif. entropy; {cutoff}\n",
                            f"R. A entropy; {rel_avg_cutoff}"])
        break