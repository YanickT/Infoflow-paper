from src.network import Network, ContraNetwork
import numpy as np
import torch
from src.dataloader import get_train_data
from src.entropies import rel_entropy, cutoff_det, diff_entropy
import matplotlib.pyplot as plt

# parameters for the network
DEPTH = 20
VARS = (0.5, 0.05)  # (1.65, 0.05)  # .9316000044345856
EPOCHS = 10

# load training data
train_data, test_data = get_train_data(bs=100)

# check for device to use
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Device: {device} with {torch.cuda.device_count()} gpus")

# initialize neural network
net = Network(28 ** 2, 10, DEPTH, VARS, device=device, size="shrinking")  # input should be 32**2 for CIFAR10 data
conet = ContraNetwork(net, device=device)

# train neural network
#loss, accs = net.training(train_data, test_data, EPOCHS)
#_, acc = net.eval(test_data)
#print(f"Accuracy of {acc} reached")

# train reconstruction
conet.train(train_data)

# create inputs which would cause a maximal output for a specific class
#fig, ax = plt.subplots(2, 5)
#axs = ax.flatten()
#for i in range(10):
#    image = conet.get_perfect(i, full=-1.0)[0].reshape(28, 28)
#    axs[i].imshow(image)
#plt.show()

# show some reconstructions and determined cutoffs
for inp, _ in test_data:
    cascades = conet.cascade(inp)
    reference = cascades[0]

    # get cutoff from cascades with relative entropy
    entropies = [rel_entropy(reference, sec).tolist() for sec in cascades[1:-2]]
    print(len(entropies), len(entropies[0]))
    entropies = tuple(zip(*entropies))

    cutoffs = cutoff_det(entropies, rtol=0.01).tolist()
    print(f"Rel. entropy - Cutoff: {np.mean(cutoffs)} $ \\pm $ {np.std(cutoffs, ddof=1)}")

    # get cutoff from cascades with differential entropy
    entropies = [[diff_entropy(sec).tolist() for sec in cascades[1:-2]]]
    cutoffs = cutoff_det(entropies, rtol=0.02).tolist()
    print(f"Dif. entropy - Cutoff: {cutoffs}")
    break