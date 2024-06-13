import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import os
import pickle

matplotlib.rc("font", **{"size": 22})


def plot_theoretic():
    with open("../theoretic_curve.txt", "rb") as doc:
        xis, test_w = pickle.load(doc)
    plt.plot(test_w, xis, "-", label="$\\xi_c$ - Mean field prediction", color="black")
    plt.plot(test_w, 6 * xis, "--", label="$6\\xi_c$  - Mean field prediction", color="black")


path = "data/collect_falling"
path2s = ["data/collect_data/data"]


files = os.listdir(path)
nets = [file for file in files if not "conet" in file]

# get only most shallow net and conet for each variance
nets_selected = {}
for net in nets:
    var = float(net.split("_")[1])
    depth = int(net.split("_")[2][:-4])
    if not (var in nets_selected):
        nets_selected[var] = (depth, net)
    elif nets_selected[var][0] > depth:
        nets_selected[var] = (depth, net)
nets = nets_selected


# plot the accuracy
norm = matplotlib.colors.Normalize(vmin=0.1, vmax=0.4)  # 0.43
cmap = matplotlib.cm.get_cmap("viridis")

# get accuracy
for key in nets:

    with open(f"{path}/{nets[key][1]}", "r") as doc:
        infos = doc.readlines()
    infos = [info.replace("\n", "") for info in infos]
    infos = [float(info.split(";")[1]) for info in infos]
    acc = max(infos)

    # plot bar with accuracy, transparent and depth
    # plot bar on top with 0.1 accuracy, transparent
    plt.bar(float(key), 200, width=0.1, color=cmap(norm(0.1)), alpha=0.6)
    plt.bar(float(key), nets[key][0], width=0.1, color=cmap(norm(acc)), alpha=0.6)



# get all cutoffs
for path2 in path2s:
    cutoffs = [[], [], []]
    conets = [file for file in os.listdir(path2)]
    for conet in conets:
        with open(f"{path2}/{conet}", "r") as doc:
            lines = doc.readlines()

        key = float(conet.split("_")[1][:-4])
        for i, line in enumerate(lines):
            print(f"{i}: {line}")
            depth = line.replace("\n", "").split(";")[1]
            cutoffs[i].append((key, float(depth)))

    # plot all cutoffs
    for index, entropy_type in zip(range(3), cutoffs):
        if not (index in [2]):  # [rel_entropy, diff_entropy, avg_rel_entropy] if [2] only plots avg_rel_entropy
            continue
        entropy_type.sort(key=lambda x: x[0])
        xs, ys = tuple(zip(*entropy_type))
        plt.plot(xs, ys, "x-", lw=2.0, label="Relative Entropy cutoff")

cbar = plt.colorbar(mappable=cm.ScalarMappable(norm=norm, cmap=cmap), ax=plt.gca(), alpha=0.6)
cbar.set_label("Accuracy", rotation=90)
plot_theoretic()
plt.legend(loc="lower right")
plt.xlabel("Variance $\\sigma_w^2$")
plt.ylabel("Depth of the Network")
plt.ylim((1, 150))
plt.xlim((0.8, 4.0))
plt.show()
