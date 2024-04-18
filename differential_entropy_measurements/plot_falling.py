import os
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm


PATH = "data/50_1_falling_MNIST"
HEIGHT = 50
SAMPLEWIDTH = 0.25


# filter files to get lowest only
files = os.listdir(PATH)
selected_files = {}
for file in files:
    var = float(file.split("_")[1])
    depth = int(file.split("_")[2][:-4])

    if not (var in selected_files):
        selected_files[var] = (depth, file)

    elif selected_files[var][0] > depth:
        selected_files[var] = (depth, file)


# load necessary file and plot accuracy
norm = matplotlib.colors.Normalize(vmin=0.1, vmax=1.)
cmap = matplotlib.cm.get_cmap("viridis")

for var in selected_files:
    with open(f"{PATH}/{selected_files[var][1]}", "r") as doc:
        lines = doc.readlines()
    line = lines[-1]
    line = line.replace("\n", "")
    line = float(line.split(";")[1])

    plt.bar(float(var), HEIGHT, width=SAMPLEWIDTH, color=cmap(norm(0.1)))
    plt.bar(float(var), selected_files[var][0], width=SAMPLEWIDTH, color=cmap(norm(line)))

plt.show()