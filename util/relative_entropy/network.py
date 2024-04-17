from typing import Tuple, Iterable, List
import torch
import torch.nn as nn
import torch.nn.functional
import numpy as np
import time
import random
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.special import kl_div


class Network:
    """
    Default network-class
    Resembles a classical discriminator network with either softmax or sigmoid output (no multi-label)
    """

    def __init__(self, input_dim: int, output_dim: int, depth: int, variances: Tuple[float, float], lr: float = 1e-4,
                 momentum: float = 5e-2, device: str = "cpu", size: str = "wide"):
        """
        Initialize a new network with given parameters
        :param input_dim: int = size of the input layer
        :param output_dim: int = size of the output layer
        :param depth: int = depth of neural network (number of hidden layers)
        :param variances: [float, float] = [weight-variance, bias-variance] for initialization of free parameters
        :param lr: float = learning rate for training
        :param device: str = Device to train on
        :param size: str = either "wide", "shrinking"
        """
        super().__init__()
        self.lr = lr
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.depth = depth
        self.variances = variances

        self.forward_sequentials = []
        if size == "shrinking":
            self.sizes = [int(e) for e in torch.ceil(torch.linspace(input_dim, output_dim, depth))]
        elif size == "wide":
            self.sizes = [input_dim] * (depth - 1) + [400, output_dim]
        elif isinstance(size, (list, tuple)):
            self.sizes = size
        else:
            raise AttributeError(f"size {size} is neither 'shrinking' nor 'wide'")
        for i in range(len(self.sizes) - 1):
            lin_layer = torch.nn.Linear(self.sizes[i], self.sizes[i + 1])
            torch.nn.init.normal_(lin_layer.weight, mean=0.0, std=(np.sqrt(variances[0] / lin_layer.out_features)))
            torch.nn.init.normal_(lin_layer.bias, mean=0.0, std=(np.sqrt(variances[1])))

            self.forward_sequentials.append(
                torch.nn.Sequential(
                    lin_layer,
                    nn.Tanh()
                )
            )

            # self.forward_sequentials[-1].double()
            self.forward_sequentials[-1].to(self.device)

        self.forward_sequential = nn.Sequential(*self.forward_sequentials + [torch.nn.Softmax(dim=1)])
        self.optimizer = torch.optim.SGD(self.forward_sequential.parameters(), lr, momentum=momentum)

    def to_cpu(self):
        """
        Shift network and all layers to cpu
        :return: void
        """
        self.device = "cpu"
        self.forward_sequential.to(self.device)

    def to_device(self, device: str):
        """
        Shift network and all layers to device
        :param device: str = device to perform calculations on
        :return: void
        """
        self.device = device
        self.forward_sequential.to(self.device)

    def collect_next_activation(self, data: torch.tensor, layer: int) -> torch.tensor:
        """
        Collect the activation after applying layer to the input
        :param data: torch.tensor = Data to propagate through the layer
        :param layer: int = number of the layer to apply data to
        :return: torch.tensor = activation after applying layer
        """
        if layer < 0: raise ValueError("No layer smaller than 0")
        self.forward_sequential.eval()
        activations = []
        with torch.no_grad():
            for inp in data:
                x = self.forward_sequential[layer](inp.to(self.device))
                activations.append(x.cpu().detach())

        return activations

    def training(self, train_data: Iterable, test_data: Iterable, its: int, verbose: bool = True) -> Tuple[
        List[float], List[float]]:
        """
        Train the network on the provided data
        :param train_data: Iterable = Training data
        :param test_data: Iterable = Test data
        :param its: int = Number of epochs to train
        :param verbose: bool = Controls information output (print)
        :return: Tuple[List[float], List[float]] = loss(epoch), acc(epoch)
        """
        losses = []
        accs = []
        t1 = time.time()
        for i in range(its):
            self.forward_sequential.train()
            for inp, out in train_data:
                # x = self.forward_sequential(inp.double().to(self.device))
                x = self.forward_sequential(inp.to(self.device))
                loss = nn.functional.cross_entropy(x, out.to(self.device))

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            # evaluate on test data
            stats = self.eval(test_data)
            losses.append(stats[0])
            accs.append(stats[1])

            if verbose: print(f"\r\tIteration {i + 1} / {its} acc: {accs[-1]}  took: {time.time() - t1}", end="")  # + \
            # f"Loss: {losses[-1]: .4f}\nAcc: {accs[-1]: .4f}\n\n")

        print("")
        return losses, accs

    def eval(self, test_data: Iterable) -> Tuple[float, float]:
        """
        Test the network on the provided data
        :param test_data: Iterable = Test data
        :return: Tuple[float, float] = loss, acc
        """
        self.forward_sequential.eval()

        losses = []
        accs = []
        with torch.no_grad():
            for inp, out in test_data:
                # propagate layer through layer
                # x = inp.double().to(self.device)
                x = inp.to(self.device)
                x = self.forward_sequential(x)

                losses.append(nn.functional.cross_entropy(x, out.to(self.device)).cpu())
                accs.append((torch.argmax(x, 1) == out.to(self.device)).float().mean().item())

        return np.mean(losses), np.mean(accs)

    def save(self, path: str):
        """
        Save current network state
        :param path: str = path to store save-file at
        :return: void
        """
        torch.save({
            "model_state_dict": self.forward_sequential.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()}, path)

    def load(self, path: str):
        """
        Load network from save-file
        :param path: str = path to save-file
        :return: void
        """
        data = torch.load(path)
        self.forward_sequential.load_state_dict(data["model_state_dict"])
        self.optimizer.load_state_dict(data["optimizer_state_dict"])

    def predict_with_activations(self, data: torch.tensor) -> Tuple[int, List[torch.tensor]]:
        """
        Calculates a prediction for given input data. Moreover, returns the activation of each layer
        :param data: torch.tensor = Input data
        :return: Tuple[int, List[torch.tensor]] = prediction, activation(layer)
        """
        self.forward_sequential.eval()
        with torch.no_grad():
            # propagate layer through layer
            x = data.to(self.device)
            forward_pass = [x]
            for forward_block in self.forward_sequentials:
                x = forward_block(x)
                forward_pass.append(x.detach())
            return torch.argmax(x).item(), forward_pass


class ContraNetwork:
    """
    Networks for reconstruction
    This class is not a cascade but holds all reconstruction networks and combines them to the different cascades when
    necessary
    """

    def __init__(self, net):
        """
        Initalize the network based on the information given at net.
        :param net: Network = network to create reconstrution networks for
        """
        self.lr = net.lr
        self.device = "cpu"
        self.length = len(net.sizes) - 1

        # construct reconstruction sequentials
        self.inverse_sequentials = []
        self.optimiers = []

        for i in range(0, self.length):
            self.inverse_sequentials.append(
                torch.nn.Sequential(
                    torch.nn.Linear(net.sizes[i + 1], net.sizes[i]),
                    nn.Tanh()
                )
            )

            self.inverse_sequentials[-1].to(self.device)
            self.optimiers.append(torch.optim.Adam(self.inverse_sequentials[-1].parameters()))

    def __len__(self):
        """
        Get length of network == number of reconstruction networks
        :return: int = number of reconstruction networks
        """
        return self.length

    def set_train_mode(self):
        """
        Set all reconstruction networks into training mode
        :return: void
        """
        for inverse_sequential in self.inverse_sequentials:
            inverse_sequential.train()

    def set_eval_mode(self):
        """
        Set all reconstruction networks into eval mode
        :return: void
        """
        for inverse_sequential in self.inverse_sequentials:
            inverse_sequential.eval()

    def train_level(self, level: int, train_data: Iterable, its: int, verbose: bool = True):
        """
        Train the reconstruction network located at level
        :param level: int = level of the reconstruction network to train
        :param train_data: Iterable = training data
        :param its: int = number of epochs to train
        :param verbose: bool = controlling if print-output is provided
        :return: void
        """
        self.set_train_mode()
        for i in range(its):
            t1 = time.time()
            random.shuffle(train_data)
            for inp, output in train_data:
                output_is = self.inverse_sequentials[level](inp.to(self.device))
                loss = nn.functional.mse_loss(output.to(self.device), output_is)
                loss.backward()
                self.optimiers[level].step()
                self.optimiers[level].zero_grad()

            if verbose: print(f"Epoch took: {time.time() - t1: .3f} s")

    def save(self, path: str):
        """
        Save state of reconstruction networks to file
        :param path: str = path to save-file
        :return: void
        """
        model_dicts = [seq.state_dict() for seq in self.inverse_sequentials]
        optimizer_dicts = [opt.state_dict() for opt in self.optimiers]
        torch.save({
            "model_state_dict": model_dicts,
            "optimizer_state_dict": optimizer_dicts}, path)

    def load(self, path: str):
        """
        Load state of reconstruction networks from save-file
        :param path: str = path to save-file
        :return: void
        """
        data = torch.load(path)
        model_dicts = data["model_state_dict"]
        optimizer_dicts = data["optimizer_state_dict"]
        for seq, model_dict in zip(self.inverse_sequentials, model_dicts):
            seq.load_state_dict(model_dict)

        for opt, opt_dict in zip(self.optimiers, optimizer_dicts):
            opt.load_state_dict(opt_dict)

    def eval_level(self, level: int, test_data: Iterable) -> float:
        """
        Evaluate mse for reconstruction network at given layer
        :param level: int = level for reconstruction network to consider
        :param test_data: Iterable = test data
        :return: float = mse for provided data and reconstruction
        """
        self.set_eval_mode()
        losses = []
        with torch.no_grad():
            for inp, output in test_data:
                output_is = self.inverse_sequentials[level](inp.to(self.device))
                losses.append(nn.functional.mse_loss(output_is, output).detach().numpy())
        return np.mean(losses)

    def cascade(self, activations):
        """
        Calculate the reconstructions for each layer of network for the provided inp data. This is done by
        combining the reconstruction networks into cascades for each depth in the forward network
        :param activations: torch.tensor = input to propagate through net and determine the reconstructions for
        :return: List[torch.tensor] = reconstructions for every layer
        """
        self.set_eval_mode()

        actis = []
        with torch.no_grad():
            for i, (layer, sequential) in enumerate(zip(activations[1:], self.inverse_sequentials)):
                # inp = layer.double().to(self.device)
                inp = layer.to(self.device)
                for seq in self.inverse_sequentials[:i + 1][::-1]:
                    inp = seq(inp)
                actis.append(inp)
        return actis


def rel_ent(first: np.array, sec: np.array) -> float:
    """
    Calculate relative entropy between first and sec
    :param first: torch.tensor = first vector of activation
    :param sec: torch.tensor = sec vector of activation
    :return: float = kl div between both vectors
    """
    first = np.maximum(first.flatten(), 0)
    sec = np.maximum(sec.flatten(), 0)

    return np.sum(kl_div(first / np.sum(first) + 1e-12, sec / np.sum(sec) + 1e-12))


def predict(net, conet, data, name):
    output, actis = net.predict_with_activations(data)
    # actis_recon = conet.predict(actis)
    actis_recon_cascade = conet.cascade(actis)

    n_plots = len(actis_recon_cascade) + 1
    square = int(np.ceil(np.sqrt(n_plots)))

    fig = plt.figure(constrained_layout=True)
    gs = GridSpec(square + 1, square, figure=fig)
    # gs = GridSpec(net.depth + 1, 2, figure=fig)

    # create sub plots as grid

    input_ax = fig.add_subplot(gs[0, 0])
    root = np.sqrt(int(data.cpu().numpy).size)
    print(root)
    input_ax.imshow(data.cpu().numpy().reshape(root, root))
    data = data.cpu().numpy()
    rel_entropies = []
    row = 0
    col = 1
    for acti in actis_recon_cascade:
        input_ax = fig.add_subplot(gs[row, col])
        input_ax.imshow(acti.cpu().numpy().reshape(28, 28))
        m = rel_ent(data, acti.cpu().numpy())

        if np.any(m > 15):  # flip happend
            acti_ = np.max(acti.cpu().numpy()) - acti.cpu().numpy()
            m = rel_ent(data, acti_)
        rel_entropies.append(m)
        col += 1
        if col == square:
            col = 0
            row += 1

    # wenn noch row frei (echt frei wird später behandelt col = 0)
    if col != 0:
        row += 1

    input_ax = fig.add_subplot(gs[row:, :])
    input_ax.plot(rel_entropies)

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(18.5, 10.5)
    fig.savefig(name, dpi=100)
    plt.close()
    return rel_entropies


def kl_div_per_layer(net, conet, data):
    output, actis = net.predict_with_activations(data)
    actis_recon_cascade = conet.cascade(actis)
    return [rel_ent(data.cpu().numpy(), acti.cpu().numpy()) for acti in actis_recon_cascade]


def get_cutoff(kl_divs, rtol=0.01):
    for index_, div in enumerate(kl_divs[::-1], start=1):
        if not (np.isclose(div, kl_divs[-1], rtol=rtol)) and not (div > 12):
            index = len(kl_divs) - index_ + 1
            return index