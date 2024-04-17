from util.differntial_entropy.network import Network, ContraNetwork
import numpy as np
import torch
from util.dataloader import get_train_data
import matplotlib.pyplot as plt

# parameters for the network
DEPTH = 9
VARS = (1.65, 0.05)
EPOCHS = 100

# load training data
train_data, test_data = get_train_data(bs=100)

# check for device to use
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Device: {device} with {torch.cuda.device_count()} gpus")

# initialize neural network
net = Network(28 ** 2, 10, DEPTH, VARS, device=device)  # input should be 32**2 for CIFAR10 data
conet = ContraNetwork(net, device=device)

# train neural network
loss, accs = net.training(train_data, test_data, EPOCHS)
_, acc = net.eval(test_data)
print(f"Accuracy of {acc} reached")

# train reconstruction
conet.train(train_data)

# create inputs which would cause a maximal output for a specific class
fig, ax = plt.subplots(2, 5)
axs = ax.flatten()
for i in range(10):
    image = conet.get_perfect(i, full=-1.0)[0].reshape(28, 28)
    axs[i].imshow(image)
plt.show()

# show some reconstructions and determined cutoffs
for inp, _ in test_data:
    images = conet.cascade(inp)

    fig, axs = plt.subplots(int(np.ceil(len(images) / 5)), 5)
    axs = axs.flatten()

    cutoff = conet.cutoff_cascade(inp)
    if not (cutoff is None):
        axs[cutoff].set_title("Cutoff")
    print(f"Cutoff at {cutoff}")
    for i in range(len(images)):
        axs[i].imshow(images[i][0].reshape(28, 28))
    plt.show()
