from typing import Tuple, Union, Sequence, Iterable, List, Optional
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import inspect
from functools import reduce
import warnings
import numpy as np
import time


def initialize(net, variances):
    todo = [l for l in net.modules() if not isinstance(l, nn.Sequential)]
    for layer in todo:
        if isinstance(layer, ResBlock):
            # the forward pass is initialized normally
            todo.append(layer.sb.modules())

            # the projection
            if not isinstance(list(layer.pb.modules())[0][0], nn.Identity):
                raise NotImplemented("Can currently only handle identity in projection")

        elif isinstance(layer, nn.Linear):
            torch.nn.init.normal_(layer.weight, mean=0.0, std=(np.sqrt(variances[0] / layer.in_features)))
            if not (layer.bias is None):
                torch.nn.init.normal_(layer.bias, mean=0.0, std=(np.sqrt(variances[1])))  # (np.sqrt(variances[1]))

        elif isinstance(layer, nn.Conv2d):
            torch.nn.init.normal_(layer.weight, mean=0.0,
                                  std=np.sqrt(variances[0] / (layer.in_channels * np.prod(layer.kernel_size)))
                                  )
            torch.nn.init.normal_(layer.bias, mean=0.0, std=(np.sqrt(variances[1])))


class Cut(nn.Module):

    def __init__(self, cut="both", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cut = cut

    def forward(self, x):
        if self.cut == "both":
            return x[:, :, 1:-1, 1:-1]
        elif self.cut == "end":
            return x[:, :, :-1, :-1]
        else:
            raise AttributeError(f"Unkown cut keyword {self.cut}")


class Channel(nn.Module):

    def __init__(self, size):
        super().__init__()
        self.size = list(size)

    def __repr__(self):
        return f"Channel({self.size})"

    def forward(self, x):
        bs = x.shape[0]
        return x.view([bs] + self.size)


class View(nn.Module):

    def __init__(self, msg="", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.msg = msg

    def forward(self, x):
        print(f"{self.msg}: {x.shape if not isinstance(x, tuple) else [e.shape for e in x]}", end="")
        return x


class CoTanh(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return torch.tanh(x) + x


class Hist(nn.Module):

    def __init__(self, msg="", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.msg = msg

    def forward(self, x):
        plt.hist(x[:, 0, 0].cpu().detach().numpy().flatten())
        plt.title(self.msg)
        plt.show()
        return x

class Unpack(nn.Module):

    def __init__(self, layer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer = layer
        self.memory = None
    def forward(self, x):
        if isinstance(x, tuple):
            self.memory = torch.zeros_like(x[1])
            return self.layer(*x)
        else:
            return self.layer(x, self.memory)


class ResBlock(nn.Module):

    def __init__(self, skip_block, projection_block):
        super().__init__()
        self.sb = skip_block
        self.pb = projection_block

    def forward(self, x):
        return torch.nn.functional.tanh(self.sb(x) + self.pb(x))


class ContraNetwork:
    """
    Networks for reconstruction
    This class is not a cascade but holds all reconstruction networks and combines them to the different cascades when
    necessary
    """

    def __init__(self, net: torch.nn.Sequential, conet: torch.nn.Sequential, device: Union[str, torch.device] = "cpu"):
        """
        Initalize the network based on the information given at net.
        :param net: Sequential = network to create reconstrution networks for
        :param device: Union[str, torch.device] = device to perform calculations on
        """
        self.net = net.to(device)

        self.length = len(conet)
        self.device = device

        self.conet = conet.to(device)
        self.optimiers = [torch.optim.Adam(layer.parameters()) if list(layer.parameters()) else None for
                          layer in conet]

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
        for layer in self.conet:
            layer.train()

    def set_eval_mode(self):
        """
        Set all reconstruction networks into eval mode
        :return: void
        """
        for layer in self.conet:
            layer.eval()

    def save(self, path: str):
        """
        Save reconstruction networks at given location
        :param path: str = path + name for file to store data in
        :return: void
        """
        model_dicts = [layer.state_dict() for layer in self.conet]
        optimizer_dicts = [None if opt is None else opt.state_dict() for opt in self.optimiers]
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
        for seq, model_dict in zip(self.conet, model_dicts):
            seq.load_state_dict(model_dict)

        for opt, opt_dict in zip(self.optimiers, optimizer_dicts):
            if opt is None:
                continue
            opt.load_state_dict(opt_dict)

    def train(self, train_data: Iterable, its: int = 1):
        """
        Train all reconstruction networks
        :param train_data: Iterable = Training data
        :param its: int = number of epochs to train
        :return: void
        """
        t1 = time.time()

        self.set_train_mode()
        indices_flag = False

        for i in range(its):
            print(f"\nEpoch: {i} / {its}")
            for counter, (inp, _) in enumerate(train_data):
                print(f"\r{counter} / {len(train_data)}", end="")
                x = inp.to(self.device)
                for f, c, opt in zip(self.net, self.conet, self.optimiers):
                    with torch.no_grad():
                        x_ = f(x)
                        if isinstance(x_, tuple):  # pooling
                            x_ = x_[0]
                            indices_flag = True

                    x_ = x_.detach()

                    if not indices_flag and (isinstance(f, ResBlock) or not isinstance(f[0], nn.Flatten)):
                        loss = torch.nn.functional.mse_loss(c(x_), x)
                        loss.backward()
                        opt.step()
                        opt.zero_grad()
                    else:
                        indices_flag = False

                    # print(f"activation : {torch.min(x)} - {torch.max(x)}")
                    # print(f"is : {torch.min(c(x_))} - {torch.max(c(x_))}")
                    # plt.imshow((x[0] - c(x_)[0]).cpu().detach().numpy().reshape((28, 28)))
                    # plt.colorbar()
                    # plt.show()

                    # loss = (torch.abs(x) * (c(x_) - x).pow(2)).mean()

                    # override old x
                    x = x_

        print(f"\nTraining done in {time.time() - t1}s")

    def eval(self, test_data: Iterable, verbose=False) -> List[float]:
        """
        Evaluate the loss of each reconstruction network for the given test data
        :param test_data: Iterable = test data
        :return: List[float] = loss of each reconstruction network
        """
        self.set_eval_mode()

        losses = [0 for i in range(self.length)]
        indices_flag = False
        with torch.no_grad():
            for inp, _ in test_data:
                x = inp.to(self.device)
                for i, (f, c) in enumerate(zip(self.net, self.conet)):
                    x_ = f(x)
                    if isinstance(x_, tuple):  # pooling
                        x_ = x_[0]
                        indices_flag = True

                    # train reconstruction
                    if not indices_flag:
                        x_pred = c(x_)
                        loss = torch.nn.functional.mse_loss(x_pred, x)
                        losses[i] += loss.item()
                    else:
                        indices_flag = False

                    x_ = x_.detach()

                    # override old x
                    x = x_

        if verbose:
            for loss, c in zip(losses, self.conet):
                print(f"{c} : {loss}")
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

        x_forward = inp.to(self.device)
        images = [x_forward.detach().cpu()]
        with torch.no_grad():
            for i, f in enumerate(self.net):
                if i >= len(self.conet):
                    break
                # forward pass through the network
                if isinstance(x_forward, tuple):
                    x_forward = x_forward[0]

                x_forward = f(x_forward)

                # reconstruction
                x_back = x_forward # .detach()
                for j in range(i, -1, -1):
                    x_back = self.conet[j](x_back)

                images.append(x_back.detach().cpu())

        print(f"Cascade done in {time.time() - t1}s")
        return images

    def cascade_to(self, inp, level=None):
        self.set_eval_mode()
        if level is None:
            level = len(self.conet)

        x_forward = inp.to(self.device)
        with torch.no_grad():
            for i, f in enumerate(self.net):
                if i >= level:
                    break
                # forward pass through the network
                if isinstance(x_forward, tuple):
                    x_forward = x_forward[0]

                x_forward = f(x_forward)

            def cascade_gen():
                x_back = yield x_forward
                while True:
                    if x_back is None:
                        x_back = x_forward
                    for j in range(i - 1, -1, -1):
                        x_back = self.conet[j](x_back)
                    x_back = yield x_back.detach().cpu()

            return cascade_gen()

class Network:
    """
    Default network-class
    Resembles a classical discriminator network with either softmax or sigmoid output (no multi-label)
    """

    def __init__(self, net, device, lr=1e-3):
        self.lr = lr
        self.device = device
        self.model = net.to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

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
                x = self.model(inp.to(self.device))
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
            print(f"\nTraining done in {time.time() - t1}s")
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

                loss = nn.functional.cross_entropy(x, out.to(self.device)).cpu()
                acc = (torch.argmax(x, 1) == out.to(self.device)).float().mean().item()
                losses.append(loss)
                accs.append(acc)

        return np.mean(losses), np.mean(accs)


if __name__ == "__main__":
    import dataloader

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    depth = 10
    variances = (1.7, 0.05)

    # Conv part with skip connections
    layers = [
        ResBlock(
            torch.nn.Sequential(
                torch.nn.Conv2d(1, 1, kernel_size=(3, 3)),
                torch.nn.Conv2d(1, 1, kernel_size=(3, 3)),
            ),
            torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(28 * 28, 576, bias=False),
                Channel((1, 24, 24))
            )),

        ResBlock(
            torch.nn.Sequential(
                torch.nn.Conv2d(1, 1, kernel_size=(3, 3)),
                torch.nn.Conv2d(1, 1, kernel_size=(3, 3)),
            ),
            torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(24 * 24, 20 * 20, bias=False),
                Channel((1, 20, 20))
            )),

        ResBlock(
            torch.nn.Sequential(
                torch.nn.Conv2d(1, 1, kernel_size=(3, 3)),
                torch.nn.Conv2d(1, 1, kernel_size=(3, 3)),
            ),
            torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(20 * 20, 16 * 16, bias=False),
                Channel((1, 16, 16))
            )),

        ResBlock(
            torch.nn.Sequential(
                torch.nn.Conv2d(1, 1, kernel_size=(3, 3)),
                torch.nn.Conv2d(1, 1, kernel_size=(3, 3)),
            ),
            torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(16 * 16, 12 * 12, bias=False),
                Channel((1, 12, 12))
            )),

        ResBlock(
            torch.nn.Sequential(
                torch.nn.Conv2d(1, 1, kernel_size=(3, 3)),
                torch.nn.Conv2d(1, 1, kernel_size=(3, 3)),
            ),
            torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(12 * 12, 8 * 8, bias=False),
                Channel((1, 8, 8))
            )),

        ResBlock(
            torch.nn.Sequential(
                torch.nn.Conv2d(1, 1, kernel_size=(3, 3)),
                torch.nn.Conv2d(1, 1, kernel_size=(3, 3)),
                torch.nn.Flatten()
            ),
            torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(8 * 8, 4 * 4, bias=False),
                # Channel((1, 20, 20))
            )
        )]

    # FC part
    layer_sizes = [16, 10]
    for l1, l2 in zip(layer_sizes, layer_sizes[1:]):
        lin_layer = torch.nn.Linear(l1, l2)
        torch.nn.init.normal_(lin_layer.weight, mean=0.0, std=(np.sqrt(variances[0] / lin_layer.in_features)))
        torch.nn.init.normal_(lin_layer.bias, mean=0.0, std=(np.sqrt(variances[1])))
        layers.append(
            nn.Sequential(
                lin_layer,
                nn.Tanh()
            )
        )

    net = nn.Sequential(*layers)
    # print(net)

    train_data, test_data = dataloader.get_train_data(64, flatten=False)
    # net(next(iter(train_data))[0])
    # exit()

    # ==================================================================================================================

    conet = nn.Sequential(
        nn.Sequential(
            nn.Flatten(),
            nn.Linear(576, 784),
            Channel([1, 28, 28])
        ),
        nn.Sequential(
            nn.Flatten(),
            nn.Linear(400, 576),
            Channel([1, 24, 24])
        ),
        nn.Sequential(
            nn.Linear(200, 400),
            nn.Tanh()
        ),
        nn.Sequential(
            nn.Linear(10, 200),
            nn.Tanh()
        ),
    )

    # net.training(train_data, test_data, its=1, verbose=True)
    conet = ContraNetwork(net, conet)
    print(conet.eval(test_data))
    conet.train(train_data, its=2)
    print(conet.eval(test_data))

    # plot reconstructions
    for inp, _ in train_data:
        cascades = conet.cascade(inp)
        for cascade in cascades:
            plt.imshow(cascade[0].reshape((28, 28)))
            plt.show()
