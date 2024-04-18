from typing import Tuple, Union, Sequence, Iterable, List
import torch
import torch.nn as nn
import inspect
import time
import numpy as np
# from scipy.stats import entropy
import matplotlib.pyplot as plt


# https://math.stackexchange.com/questions/1804805/how-is-the-entropy-of-the-normal-distribution-derived
# https://de.wikipedia.org/wiki/Differentielle_Entropie
NORMALFACTOR = (0.5 * np.log(2 * np.pi * np.e))


class Network:
    """
    Default network-class
    Resembles a classical discriminator network with either softmax or sigmoid output (no multi-label)
    """

    def __init__(self, input_dim: int, output_dim: int, depth: int, variances: Tuple[float, float], lr: float = 1e-3,
                 device: Union[str, torch.device] = "cpu", size: Union[Sequence, str] = "wide"):
        """
        Initialize a new network with given parameters
        :param input_dim: int = size of the input layer
        :param output_dim: int = size of the output layer
        :param depth: int = depth of neural network (number of hidden layers)
        :param variances: [float, float] = [weight-variance, bias-variance] for initialization of free parameters
        :param lr: float = learning rate for training
        :param device: Union[str, torch.device] = Device to train on
        :param size: Union[Sequence, str] = either "wide", "shrinking" or a sequence of layer sizes
        If sequence is provided, depth will be ignored
        """
        super().__init__()
        self.lr = lr
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.depth = depth
        self.variances = variances

        layers = []
        if size == "wide":
            self.sizes = [input_dim] * (depth - 1) + [400, output_dim]
        elif size == "shrinking":
            self.sizes = [int(e) for e in torch.ceil(torch.linspace(input_dim, output_dim, depth))]
        elif isinstance(size, (list, tuple)):
            self.sizes = size
        else:
            raise AttributeError(f"Can not handle size {size}")

        for i in range(len(self.sizes) - 1):
            lin_layer = torch.nn.Linear(self.sizes[i], self.sizes[i + 1])
            torch.nn.init.normal_(lin_layer.weight, mean=0.0, std=(np.sqrt(variances[0] / lin_layer.out_features)))
            torch.nn.init.normal_(lin_layer.bias, mean=0.0, std=(np.sqrt(variances[1])))

            layers.append(lin_layer)
            layers.append(nn.Tanh())

        if self.sizes[-1] == 1:
            layers.append(torch.nn.Sigmoid())
        else:
            layers.append(torch.nn.Softmax(dim=1))
        self.model = nn.Sequential(*layers)
        self.model.to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr)

    def training(self, train_data: Iterable, test_data: Iterable, its: int) -> Tuple[List[float], List[float]]:
        """
        Train network on provided training data. Subsequent testing on test data for each epoch
        :param train_data: Iterable = data to train on
        :param test_data: Iterable = data to test with
        :param its: int = number of epochs to train
        :return: Tuple[List[float], List[float]] = loss(epoch), accuracy(epoch)
        """
        losses = []
        accs = []
        t1 = time.time()
        for i in range(its):
            for inp, out in train_data:
                # x = self.forward_sequential(inp.double().to(self.device))
                x = self.model(inp.to(self.device))

                if self.sizes[-1] == 1:
                    loss = nn.functional.binary_cross_entropy(x, out.to(self.device))
                    # print(x, out, loss)
                else:
                    loss = nn.functional.cross_entropy(x, out.to(self.device))

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            # evaluate on test data
            stats = self.eval(test_data)
            losses.append(stats[0])
            accs.append(stats[1])

            print(f"\rTraining {(i + 1)} / {its}: {time.time() - t1} with acc: {stats[1]: .3f}", end="")

        print(f"\rTraining done in {time.time() - t1}s")
        return losses, accs

    def eval(self, test_data: Iterable) -> Tuple[float, float]:
        """
        Test network on provided test data
        :param test_data: Iterable = data to test with
        :return: Tuple[float, float] = loss, accuracy
        """
        self.model.eval()

        losses = []
        accs = []
        with torch.no_grad():
            for inp, out in test_data:
                # propagate layer through layer
                # x = inp.double().to(self.device)
                x = inp.to(self.device)
                x = self.model(x)

                if self.sizes[-1] == 1:
                    loss = nn.functional.binary_cross_entropy(x, out.to(self.device)).cpu()
                    acc = (torch.round(x) == out.to(self.device)).float().mean().item()
                else:
                    loss = nn.functional.cross_entropy(x, out.to(self.device)).cpu()
                    acc = (torch.argmax(x, 1) == out.to(self.device)).float().mean().item()
                losses.append(loss)
                accs.append(acc)

        return np.mean(losses), np.mean(accs)


class ContraNetwork:
    """
    Networks for reconstruction
    This class is not a cascade but holds all reconstruction networks and combines them to the different cascades when
    necessary
    """

    def __init__(self, net: Network, device: Union[str, torch.device] = "cpu"):
        """
        Initalize the network based on the information given at net.
        :param net: Network = network to create reconstrution networks for
        :param device: Union[str, torch.device] = device to perform calculations on
        """
        self.net = net
        self.lr = net.lr
        self.device = device
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

    def __len__(self) -> int:
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

    def save(self, path: str):
        """
        Save reconstruction networks at given location
        :param path: str = path + name for file to store data in
        :return: void
        """
        model_dicts = [seq.state_dict() for seq in self.inverse_sequentials]
        optimizer_dicts = [opt.state_dict() for opt in self.optimiers]
        torch.save({
            "model_state_dict": model_dicts,
            "optimizer_state_dict": optimizer_dicts}, path)

    def load(self, path: str):
        """
        Load reconstruction networks from file at given path
        :param path: str = path to file to load
        :return: void
        """
        data = torch.load(path)
        model_dicts = data["model_state_dict"]
        optimizer_dicts = data["optimizer_state_dict"]
        for seq, model_dict in zip(self.inverse_sequentials, model_dicts):
            seq.load_state_dict(model_dict)

        for opt, opt_dict in zip(self.optimiers, optimizer_dicts):
            opt.load_state_dict(opt_dict)

    def _hijack_network(self):
        """
        Analyse network-class, find sequence of layers and hijack them
        """
        # check if network has a sequential as attribute
        sequentials = [(name, value) for name, value in inspect.getmembers(self.net) if
                       isinstance(value, nn.Sequential)]
        if len(sequentials) == 1:
            # print(f"Found sequence of layers in '{sequentials[0][0]}'")
            # split into set of sequentials
            seq_list = []
            temp = []
            for layer in sequentials[0][1]:
                if isinstance(layer, nn.Tanh):
                    temp.append(layer)
                    seq_list.append(nn.Sequential(*temp))
                    seq_list[-1].eval()
                    seq_list[-1].to(self.device)
                    temp = []
                else:
                    temp.append(layer)

            return seq_list
        elif len(sequentials) > 1:
            raise AttributeError("Net has more than one sequential. Could not detect layers")

        # check if network has forward pass
        candidates = [(name, value) for name, value in inspect.getmembers(self.net) if
                      inspect.ismethod(value) and name == "forward"]
        if len(candidates) == 1:
            print(f"Found forward implementation of network")
            raise NotImplementedError("Not yet implemented")
        else:
            raise AttributeError(f"Could not detect a layer sequence to hijack")

    def train(self, train_data: Iterable, its: int = 1):
        """
        Train all reconstruction networks
        :param train_data: Iterable = Training data
        :param its: int = number of epochs to train
        :return: void
        """
        t1 = time.time()

        self.set_train_mode()
        layers = self._hijack_network()
        for i in range(its):
            for inp, _ in train_data:
                x = inp.to(self.device)
                for f, c, opt in zip(layers, self.inverse_sequentials, self.optimiers):
                    with torch.no_grad():
                        x_ = f(x)

                    # train reconstruction
                    x_ = x_.detach()
                    x_pred = c(x_)
                    loss = torch.nn.functional.mse_loss(x_pred, x)
                    loss.backward()
                    opt.step()
                    opt.zero_grad()

                    # override old x
                    x = x_
        print(f"Training done in {time.time() - t1}s")

    def eval(self, test_data: Iterable) -> List[float]:
        """
        Evaluate the loss of each reconstruction network for the given test data
        :param test_data: Iterable = test data
        :return: List[float] = loss of each reconstruction network
        """
        self.set_eval_mode()
        layers = self._hijack_network()
        losses = [0 for i in range(len(layers))]
        with torch.no_grad():
            for inp, _ in test_data:
                x = inp.to(self.device)
                for i, (f, c) in enumerate(zip(layers, self.inverse_sequentials)):
                    x_ = f(x)

                    # train reconstruction
                    x_ = x_.detach()
                    x_pred = c(x_)
                    loss = torch.nn.functional.mse_loss(x_pred, x)
                    losses[i] += loss.item()

                    # override old x
                    x = x_

        return losses

    def cascade(self, inp: torch.tensor) -> List[torch.tensor]:
        """
        Calculate the reconstructions for each layer of network for the provided inp data. This is done by
        combining the reconstruction networks into cascades for each depth in the forward network
        :param inp: torch.tensor = input to propagate through net and determine the reconstructions for
        :return: List[torch.tensor] = reconstructions for every layer
        """
        t1 = time.time()

        self.set_eval_mode()

        layers = self._hijack_network()
        x_forward = inp.to(self.device)
        images = [x_forward.detach().cpu()]
        with torch.no_grad():
            for i, f in enumerate(layers):
                # forward pass through the network
                x_forward = f(x_forward)

                # reconstruction
                x_back = x_forward.detach()
                for j in range(i, -1, -1):
                    x_back = self.inverse_sequentials[j](x_back)
                images.append(x_back.detach().cpu())

        print(f"Cascade done in {time.time() - t1}s")
        return images

    def cutoff_cascade(self, inp: torch.tensor, cutoff_threshold: float = -5.5) -> int:
        """
        Determine cutoff using the cascades and reconstructions
        :param inp: torch.tensor = torch.tensor = input to propagate through net (should be batch with size > 10)
        :param cutoff_threshold: float = threshold value for information cutoff
        :return: int = information cutoff depth
        """
        self.set_eval_mode()

        layers = self._hijack_network()
        x_forward = inp.to(self.device)

        #root = int(np.ceil(np.sqrt(self.length))) + 1
        #fig, axs = plt.subplots(root, root)
        #axs = axs.flatten()
        #ax_counter = 0
        #axs[ax_counter].imshow(inp, aspect="auto")
        #ax_counter += 1

        with torch.no_grad():
            for i, f in enumerate(layers):
                # forward pass through the network
                x_forward = f(x_forward)

                # reconstruction
                x_back = x_forward.detach()
                for j in range(i, -1, -1):
                    x_back = self.inverse_sequentials[j](x_back)

                #axs[ax_counter].imshow(x_back, aspect="auto")
                #ax_counter += 1

                # compare images and find difference
                # std = torch.sum(torch.std(x_back, dim=0))
                #entropy_ = 0
                #for k in range(x_back.shape[1]):
                #    c, _ = np.histogram(x_back[:, k], bins=20, range=(-1, 1))
                #    entropy_ += entropy(c)
                std = torch.std(x_back, dim=0)
                # print(NORMALFACTOR + torch.log(std))
                entropy_ = torch.mean(NORMALFACTOR + torch.log(std))
                # print(entropy_)
                if entropy_ <= cutoff_threshold:
                    # plt.show()
                    return i

        # plt.show()
        return self.length

    def get_perfect(self, index: int, full=0.0) -> torch.tensor:
        """
        Artificially create reconstruction for a maximal identification of a single class
        :param index: int = number of class to create reconstruction for
        :param full: float = activation for the not-wanted-classes
        :return: torch.tensor = artificially created reconstruction
        """
        inp = torch.full((1, self.net.sizes[-1]), full)
        inp[0, index] = 1
        inp = inp.to(self.device)
        for c in self.inverse_sequentials[::-1]:
            inp = c(inp)
        return inp.detach().cpu()

