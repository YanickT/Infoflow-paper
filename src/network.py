from typing import Tuple, Union, Sequence, Iterable, List, Optional
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import inspect
from functools import reduce
import warnings
import numpy as np
import time


class Channel(nn.Module):

    def __init__(self, size):
        super().__init__()
        self.size = list(size)

    def __repr__(self):
        return f"Channel({self.size})"

    def forward(self, x):
        bs = x.shape[0]
        return x.view([bs] + self.size)


class ResBlock(nn.Module):

    def __call__(self, skip_block, projection_block):
        self.sb = skip_block
        self.pb = projection_block

    def forward(self, x):
        return self.sb(x) + self.pb(x)


def conv_sizes(input_dim, conv_layers):
    # process input dimension
    if isinstance(input_dim, List):
        dim = len(input_dim)
    else:
        dim = 1
        input_dim = [input_dim]

    # check if each layer has correct dimension
    if not all([len(conv.kernel_size) == dim for conv in conv_layers]):
        raise AttributeError("At least one Convolutional layer has wrong dimensions!")

    # calculate sizes for each layer
    sizes = []
    size = [conv_layers[0].in_channels] + input_dim  # batch-size is excluded
    sizes.append(size[:])

    for conv in conv_layers:
        if conv.padding == "same":
            continue

        elif conv.padding == "valid":
            conv.padding = [0] * dim

        new_size = [conv.out_channels]
        for d in range(dim):
            width = int((size[d + 1] + 2 * conv.padding[d] - conv.dilation[d] * (conv.kernel_size[d] - 1) - 1) /
                        conv.stride[d] + 1)  # + 1 in size[d + 1] for in_channel in size
            new_size.append(width)
        size = new_size
        sizes.append(size[:])

    return sizes


class Network:
    """
    Default network-class
    Resembles a classical discriminator network with either softmax or sigmoid output (no multi-label)
    """

    def __init__(self, input_dim: Union[int, List[int]], output_dim: int, depth: int,
                 variances: Tuple[float, float], lr: float = 1e-3,
                 device: Union[str, torch.device] = "cpu", size: Union[Sequence, str] = "wide",
                 convolutions: Optional[List] = None):
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
        self.lr = lr
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.depth = depth
        self.variances = variances

        # calculate input size of linear part and prepare convs list
        convs = []
        if not (convolutions is None):
            input_dim = conv_sizes(input_dim, convolutions)[-1]
            input_dim = reduce(lambda x, y: x * y, input_dim)

            # prepare convolutional layers
            for conv in convolutions:
                convs.append(conv)
                convs.append(nn.Tanh())
            convs.append(nn.Flatten())

        elif isinstance(input_dim, List):
            # if no convolution is involved, we flatten the input directly
            input_dim = reduce(lambda x, y: x * y, input_dim)

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
            torch.nn.init.normal_(lin_layer.weight, mean=0.0, std=(np.sqrt(variances[0] / lin_layer.in_features)))
            torch.nn.init.normal_(lin_layer.bias, mean=0.0, std=(np.sqrt(variances[1])))

            layers.append(lin_layer)
            layers.append(nn.Tanh())

        if self.sizes[-1] == 1:
            layers.append(torch.nn.Sigmoid())
        else:
            layers.append(torch.nn.Softmax(dim=1))
        self.model = nn.Sequential(*(convs + layers))
        self.model.to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr)

    def training(self, train_data: Iterable, test_data: Iterable, its: int, verbose: bool = False) -> Tuple[
        List[float], List[float]]:
        """
        Train network on provided training data. Subsequent testing on test data for each epoch
        :param train_data: Iterable = data to train on
        :param test_data: Iterable = data to test with
        :param its: int = number of epochs to train
        :param verbose: bool = print information
        :return: Tuple[List[float], List[float]] = loss(epoch), accuracy(epoch)
        """
        losses = []
        accs = []
        t1 = time.time()
        for i in range(its):
            self.model.train()
            for inp, out in train_data:
                # x = self.forward_sequential(inp.double().to(self.device))
                x = self.model(inp.to(self.device))

                if self.sizes[-1] == 1:
                    loss = nn.functional.binary_cross_entropy(x, out.to(self.device))
                else:
                    loss = nn.functional.cross_entropy(x, out.to(self.device))

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            # evaluate on test data
            stats = self.eval(test_data)
            losses.append(stats[0])
            accs.append(stats[1])

            if verbose:
                print(f"\rTraining {(i + 1)} / {its}: {time.time() - t1} with acc: {stats[1]: .3f}", end="")

        if verbose:
            print(f"\n\rTraining done in {time.time() - t1}s")
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

    def __init__(self, net: Network, device: Union[str, torch.device] = "cpu", convTrans=False):
        """
        Initalize the network based on the information given at net.
        :param net: Network = network to create reconstrution networks for
        :param device: Union[str, torch.device] = device to perform calculations on
        """
        self.net = net
        self.device = device

        layers = self._hijack_network()
        self.length = len(layers)

        # construct reconstruction sequentials
        self.inverse_sequentials = []
        self.optimiers = []

        conv_layer = [seq[0] for seq in layers if isinstance(seq[0], (nn.Conv1d, nn.Conv2d, nn.Conv3d))]
        if conv_layer:
            sizes = conv_sizes(net.input_dim, conv_layer)
        else:
            sizes = []
        prod = lambda x: reduce(lambda x, y: x * y, x)
        for i, seq in enumerate(layers):
            no_opti_flag = False
            if isinstance(seq[0], nn.Linear):
                self.inverse_sequentials.append(
                    torch.nn.Sequential(
                        nn.Linear(seq[0].out_features, seq[0].in_features),
                        nn.Tanh()
                    )
                )
            elif isinstance(seq[0], (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                if convTrans:
                    warnings.warn("Currently only 2d ConvTrans is supported")
                    self.inverse_sequentials.append(
                        nn.Sequential(
                            nn.ConvTranspose2d(seq[0].out_channels,
                                               seq[0].in_channels,
                                               seq[0].kernel_size,
                                               stride=seq[0].dilation,
                                               dilation=seq[0].stride),
                            nn.Tanh()
                        )
                    )
                else:
                    self.inverse_sequentials.append(
                        nn.Sequential(
                            nn.Flatten(),
                            nn.Linear(prod(sizes[i + 1]), prod(sizes[i])),
                            nn.Tanh(),
                            Channel(sizes[i])
                        )
                    )
            elif isinstance(seq[0], nn.Flatten):
                self.inverse_sequentials.append(
                    torch.nn.Sequential(
                        Channel(sizes[i])
                    )
                )
                no_opti_flag = True
            else:
                raise AttributeError(f"Could not identify layer {seq[0]} or Layer-type is not supported")

            self.inverse_sequentials[-1].to(self.device)
            if not no_opti_flag:
                self.optimiers.append(torch.optim.Adam(self.inverse_sequentials[-1].parameters()))
            else:
                self.optimiers.append(None)

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

    def _group_members(self, members):
        # split into set of sequentials
        seq_list = []
        temp = []
        for layer in members:
            if isinstance(layer, (nn.Tanh, nn.Flatten, nn.ReLU)):
                temp.append(layer)
                seq_list.append(nn.Sequential(*temp))
                seq_list[-1].eval()
                seq_list[-1].to(self.device)
                temp = []
            else:
                temp.append(layer)

        return seq_list

    def _hijack_network(self):
        """
        Analyse network-class, find sequence of layers and hijack them
        """
        # check if network has a sequential as attribute
        members = [(name, value) for name, value in inspect.getmembers(self.net) if
                   isinstance(value, nn.Sequential)]
        if len(members) > 1:
            raise AttributeError("Net has more than one sequential. Could not detect layers")
        elif len(members) == 1:
            return self._group_members(members[0][1])
        else:
            raise AttributeError("Net has no sequential... Sorry")

    def get_activations(self, data: Iterable) -> List[List[torch.tensor]]:
        layers = self._hijack_network()
        activations = []
        with torch.no_grad():
            for inp, _ in data:
                temp = [inp.to(self.device)]
                for f in layers:
                    temp.append(f(temp[-1]))
                activations.append(temp)
        return activations

    def get_activations_sorted(self, data: Iterable) -> List[List[torch.tensor]]:
        layers = self._hijack_network()
        activations = [[] for i in range(len(conet) + 1)]
        with torch.no_grad():
            for inp, _ in data:
                activations[0].append(inp.to(self.device))
                for j, f in enumerate(layers, start=1):
                    activations[j].append(f(activations[j - 1][-1]))
        return activations

    def train_lvl(self, lvl: int, train_data: Iterable, its: int = 1):
        c = self.inverse_sequentials[lvl]
        opt = self.optimiers[lvl]
        t1 = time.time()
        for _ in range(its):
            for inp, out in train_data:
                pred = c(inp)
                loss = torch.nn.functional.mse_loss(pred, out)
                loss.backward()
                opt.step()
                opt.zero_grad()
        print(f"Training done in {time.time() - t1}s")

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
            for counter, (inp, _) in enumerate(train_data):
                print(f"\r{counter} / {len(train_data)}", end="")
                x = inp.to(self.device)
                for f, c, opt in zip(layers, self.inverse_sequentials, self.optimiers):
                    with torch.no_grad():
                        x_ = f(x)

                    # train reconstruction
                    # print(f"Use {f}: {x.shape} -> {x_.shape}")
                    # print(f"With inverse {c}:", end="")
                    x_ = x_.detach()
                    x_pred = c(x_)
                    # print(f"{x_.shape} -> {x_pred.shape}")
                    # print("\n\n")

                    if not (len(c) == 1 and isinstance(c[0], Channel)):
                        loss = torch.nn.functional.mse_loss(x_pred, x)
                        loss.backward()
                        opt.step()
                        opt.zero_grad()

                    # override old x
                    x = x_
        print(f"\nTraining done in {time.time() - t1}s")

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


if __name__ == "__main__":
    import dataloader

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    net = Network([28, 28], 10, 10, (1.76, 0.05), convolutions=[nn.Conv2d(1, 3, (5, 5), padding="valid")])
    train_data, test_data = dataloader.get_train_data(64, flatten=False)
    # net.training(train_data, test_data, its=1, verbose=True)
    conet = ContraNetwork(net, convTrans=False)
    print(conet.eval(test_data))
    conet.train(train_data, its=1)
    print(conet.eval(test_data))

    # plot reconstructions
    for inp, _ in train_data:
        cascades = conet.cascade(inp)
        for cascade in cascades:
            plt.imshow(cascade[0].reshape((28, 28)))
            plt.show()
