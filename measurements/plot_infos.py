import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rc("font", **{"size": 22})

PATH = "data/collect_data/entropy"

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

im = plt.imshow(table, aspect="auto", origin='lower', cmap="Spectral")  # coolwarm, hot, bwr, seismic
cbar = plt.colorbar(im)
cbar.set_label("Differential entropy", rotation=90)
plt.xlabel("Variance $\\sigma_w^2$")
plt.ylabel("Depth of the Network")
plt.xticks(list(range(len(vars_)))[::10], vars_[::10])
plt.grid()
plt.show()
