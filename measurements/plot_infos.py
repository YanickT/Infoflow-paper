import os
import numpy as np
import  matplotlib
import matplotlib.pyplot as plt

# Do it on FashionMNIST, CIFAR100

matplotlib.rc("font", **{"size": 22})


PATH = (f"data/convnet_depth=400/entropy")
# PATH = (f"data/resnet/entropy")


files = os.listdir(PATH)
table = []
for file in files:
    with open(f"{PATH}/{file}", "r") as doc:
        lines = doc.readlines()
    values = [float(v) for v in lines[0].split(";")]
    var_ = float(file.split("_")[-1][:-4])
    table.append((var_, values))

table.sort(key=lambda x: x[0])
vars_ = [t[0] for t in table]
table = np.array([t[1] for t in table])
table = np.rot90(table)
table = table[::-1]


# trainables = np.array([atl.get_trainables()])
table = np.clip(table, a_min=-10, a_max=0)
# For CNN the information loss in the chaotic phase is smaller, clip (a_min = -10) to make it visible
# For ResNet a_min=-19 allows to see the different phases nicely.


loss_file_path = "/".join([s for s in PATH.split("/")[:-1]])
if os.path.isfile(f"{loss_file_path}/loss.csv"):
    fig, axs = plt.subplots(2, 1, gridspec_kw={"height_ratios": [1, 20]}, sharex=True)

    with open(f"{loss_file_path}/loss.csv", "r") as doc:
        lines = doc.readlines()[0].split(";")
    accs = [float(line.split(", ")[1][:-1]) for line in lines]
    trainables = [[2 if e > 0.15 else 1 if e > 0.1 else 0 for e in accs]]
    axs = axs[::-1]
    axs[1].imshow(trainables, aspect="auto", cmap=matplotlib.colors.ListedColormap(['white', 'grey', 'black']))
    axs[1].set_title("Trainability")
    axs[1].set_yticks([])
    plt.subplots_adjust(hspace=0.016)
else:
    fig, axs = plt.subplots(1, 1)
    axs = [axs]

# entropy image
axs[0].set_xlabel("Variance $\\sigma_w^2$")
axs[0].set_xticks(list(range(len(vars_)))[::50], vars_[::50])  #
axs[0].set_ylabel("Depth of the Network")
im = axs[0].imshow(table, aspect="auto", origin='lower', cmap="Spectral")

# add colobar
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.1, 0.02, 0.8])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label("Differential entropy", rotation=90)

# plt.grid()
plt.show()