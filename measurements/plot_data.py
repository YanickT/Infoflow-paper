import os
import matplotlib.pyplot as plt

PATH = "data_120"

files = os.listdir(PATH)
files.sort(key=lambda x: float(x.split("_")[1][:-4]))

xs = []
rel_entropies = []
dif_entropies = []
for file in files:
    with open(f"{PATH}/{file}", "r") as doc:
        lines = doc.readlines()
    xs.append(float(file.split("_")[1][:-4]))
    rel_entropies.append(float(lines[0].split(";")[1]))
    dif_entropies.append(int(lines[1].split(";")[1]))

plt.plot(xs, rel_entropies, label="Rel. entropy")
plt.plot(xs, dif_entropies, label="Dif. entropy")
plt.grid()
plt.legend()
plt.show()