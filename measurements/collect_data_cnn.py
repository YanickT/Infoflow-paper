import src.dataloader as dataloader
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import os
from src.entropies import rel_entropy, diff_entropy, cutoff_det
from src.network_modified import ResBlock, Channel, ContraNetwork, View, initialize, Network
from src.network_types import get_conv_net


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Training with {device}")
variances_ = np.arange(0.1, 6.0, 0.01)
bs = 100
train_data, test_data = dataloader.get_train_data(bs, flatten=False)

PATH = f"convnet_depth=400_fine"
os.mkdir(PATH)
os.mkdir(f"{PATH}/entropy")
os.mkdir(f"{PATH}/rel_entropy")
os.mkdir(f"{PATH}/data")
os.mkdir(f"{PATH}/images")


for var_w in variances_:
    variances = (var_w, 2e-5)
    print(variances)

    layers, layer_conet = get_conv_net(400, c=128)
    # ==================================================================================================================
    net = nn.Sequential(*layers)
    initialize(net, variances)  # normal initialize every layer in net
    fonet = Network(net, device=device)
    #print(f"Training {var_w}")
    #fonet.training(train_data, test_data, its=1, verbose=True)

    conet = nn.Sequential(*layer_conet)

    conet = ContraNetwork(net, conet, device=device)

    #print(conet.eval(test_data))
    conet.train(train_data, its=1)
    # fonet.training(train_data, test_data, its=1, verbose=True)
    #print(conet.eval(test_data))


    """for inp, _ in test_data:
        cascades = conet.cascade(inp)
        reference = cascades[0]
        plt.imshow(reference[0].cpu().detach().numpy().reshape((28, 28)))
        plt.show()
        for i in range(len(conet)):
            plt.imshow(cascades[i][0].cpu().detach().numpy().reshape((28, 28)))
            plt.show()"""


    for inp, _ in test_data:
        cascades = conet.cascade(inp)
        reference = cascades[0]

        """plt.imshow(reference[0].cpu().detach().numpy().reshape((28, 28)))
        plt.show()
        for i in range(1, len(conet)):
           plt.imshow(cascades[i][0].cpu().detach().numpy().reshape((28, 28)))
           plt.show()"""

        entropies = [rel_entropy(reference.reshape(bs, 28 * 28), sec.reshape(bs, 28 * 28)).tolist() for sec in
                     cascades[1:-2]]
        entropies = np.array(tuple(zip(*entropies)))


        # determine cutoff of average
        # entropies_avg = entropies / entropy(reference.numpy().transpose())[:, None]
        entropies_avg = entropies / entropies[:, -1][:, None]  # np.max(entropies, axis=1)[:, None]
        avg_entropies = np.median(entropies_avg.transpose(), axis=1).reshape([1, len(entropies[0])])
        with open(f"{PATH}/rel_entropy/entropy_{var_w : .3f}.csv", "w") as doc:
            doc.writelines("\n".join([";".join([str(v) for v in avg_entropies.flatten()])]))

        # avg_entropies = (entropies_avg.transpose().prod(axis=1)**(1/len(entropies[0]))) .reshape([1, len(entropies[0])])
        rel_avg_cutoff = cutoff_det(avg_entropies, rtol=0.01)[0]
        print(f"Rel. Avg entropy - Cutoff: {rel_avg_cutoff}")

        # cutoffs = cutoff_det(entropies, atol=5e-2, method="differential").tolist()
        cutoffs = cutoff_det(entropies, rtol=0.01).tolist()
        rel_cutoff = (np.mean(cutoffs), np.std(cutoffs, ddof=1))
        print(f"Rel. entropy - Cutoff: {rel_cutoff[0]} $ \\pm $ {rel_cutoff[1]}")

        # get cutoff from cascades with differential entropy
        entropies = [[diff_entropy(sec.reshape(bs, 28 * 28)).tolist() for sec in cascades[1:-2]]]
        with open(f"{PATH}/entropy/diff_entropy_{var_w : .3f}.csv", "w") as doc:
            doc.writelines("\n".join([";".join([str(v) for v in e]) for e in entropies]))
        # plt.plot(entropies[0])
        # plt.show()
        # cutoff = cutoff_det(entropies, atol=5e-1, method="differential")[0]
        cutoff = cutoff_det(entropies, rtol=0.01, method="differential")[0]
        print(f"Dif. entropy - Cutoff: {cutoff}")

        with open(f"{PATH}/data/entropy_{var_w : .3f}.csv", "w") as doc:
            doc.writelines([f"Rel. entropy; {rel_cutoff[0]}; {rel_cutoff[1]}\n",
                            f"Dif. entropy; {cutoff}\n",
                            f"R. A entropy; {rel_avg_cutoff}"])
        break

    # plot reconstructions
    # for inp, _ in train_data:
    #    cascades = conet.cascade(inp)
    #    for cascade in cascades:
    #        plt.imshow(cascade[0].reshape((28, 28)))
    #        plt.show()
