import pickle
import matplotlib.pyplot as plt
import matplotlib
import os


PATH = "data/10_wide_10eps_MNIST"
SAMPLESTEP = 0.5
MAXHEIGHT = 10


# get files
files = os.listdir(PATH)

# split into conet files and net files
conets = [file for file in files if "conet" in file]
nets = [file for file in files if not ("conet" in file)]

# get accuracy of net files
# get some colormap to map accuracy to color
norm = matplotlib.colors.Normalize(vmin=0.1, vmax=1.0)  # 0.43 for CIFAR10 recommended
cmap = matplotlib.cm.get_cmap("viridis")

for net in nets:
    # read variance from filename
    var_w = float(net.split("_")[1][:-4])

    # open file and read accuracy
    with open(f"{PATH}/{net}", "r") as doc:
        line = doc.readlines()[-1]
    acc = float(line.split(";")[1])

    # create a bar plot of that
    plt.bar(var_w, MAXHEIGHT, width=SAMPLESTEP, color=cmap(norm(acc)))


# get cutoff depth
cutoffs = {}
vars_w = []
print(conets)
for conet in conets:
    # read variance from filename
    vars_w.append(float(conet.split("_")[1][:-4]))

    # open file and read cutoffs
    with open(f"{PATH}/{conet}", "r") as doc:
        lines = doc.readlines()
    lines = [line.replace("\n", "").split(";") for line in lines]
    lines = [(float(f), float(s)) for f, s in lines]
    for f, s in lines:
        if not (f in cutoffs):
            cutoffs[f] = []
        cutoffs[f].append(s)

for threshold in cutoffs:
    plt.plot(vars_w, cutoffs[threshold], label=f"threshold: {threshold}")

plt.legend()
plt.grid()
plt.show()

