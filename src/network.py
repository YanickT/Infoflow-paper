from typing import Tuple, Union, Sequence, Iterable, List, Optional
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import inspect
from itertools import repeat
from functools import reduce
import torch.multiprocessing as mpl
import numpy as np


def worker(c: torch.nn.Sequential, opt: torch.optim.Optimizer, train_data: Tuple[Tuple[torch.tensor, torch.tensor]],
           its: int):
    for _ in range(its):
        for inp, out in train_data:
            loss = torch.nn.functional.mse_loss(c(inp), out)
            loss.backward()
            opt.step()
            opt.zero_grad()


def queue_worker(q):
    while True:
        args = q.get()
        if args == "DONE":
            break

        c, opt, train_data, its = args
        for _ in range(its):
            for inp, out in train_data:
                loss = torch.nn.functional.mse_loss(c(inp), out)
                loss.backward()
                opt.step()
                opt.zero_grad()


class Network:
    """
    Default network-class
    Resembles a classical discriminator network with either softmax or sigmoid output (no multi-label)
    """

    def __init__(self, input_dim: Union[int, List[int]], output_dim: int, depth: int,
                 variances: Tuple[float, float], lr: float = 1e-3,
                 device: Union[str, torch.device] = "cpu", size: Union[Sequence, str] = "wide", convolutions: Optional[List]=None):
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

        new_convs = []
        if not (convolutions is None):
            if isinstance(input_dim, List):
                dim = len(input_dim)
            else:
                dim = 1
                input_dim = [input_dim]

            if not all([len(conv.kernel_size) == dim for conv in convolutions]):
                raise AttributeError("At least one Convolutional layer has wrong dimensions!")

            size = [convolutions[0].in_channels] + input_dim  # batch-size is excluded
            for conv in convolutions:
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
                new_convs.append(conv)
                new_convs.append(nn.Tanh())

            new_convs.append(nn.Flatten())
            input_dim = reduce(lambda x, y: x * y, size)

        elif isinstance(input_dim, List):
            # if no convolution is involved, we flatten the input directly
            input_dim = np.sum(input_dim)

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
        self.model = nn.Sequential(*(new_convs + layers))
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

    def __init__(self, net: Network, device: Union[str, torch.device] = "cpu", lr=1e-3):
        """
        Initalize the network based on the information given at net.
        :param net: Network = network to create reconstrution networks for
        :param device: Union[str, torch.device] = device to perform calculations on
        """
        self.net = net
        self.lr = lr
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

    def _group_members(self, members):
        # split into set of sequentials
        seq_list = []
        temp = []
        for layer in members:
            if isinstance(layer, nn.Tanh):
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

    def train_parallel_sequential(self, data: Iterable, its=1, stop=5):
        layers = self._hijack_network()
        first = [inp.to(self.device) for inp, _ in data]
        stop = len(self) - stop
        with mpl.Pool() as pool:
            for i, f in enumerate(layers):
                # get all new activations
                with torch.no_grad():
                    sec = [f(e) for e in first]

                # start process of training corresponding reconstruction
                if i < stop:
                    p = pool.apply_async(worker,
                                         (self.inverse_sequentials[i], self.optimiers[i], tuple(zip(sec, first)), its))
                else:
                    worker(self.inverse_sequentials[i], self.optimiers[i], tuple(zip(sec, first)), its)
                first = sec
                del sec

            del first
            p.wait(timeout=10)

    def train_queue(self, data, its=1, cores=mpl.cpu_count()):
        layers = self._hijack_network()
        first = [inp.to(self.device) for inp, _ in data]

        # creat queue
        q = mpl.Queue()

        # start workers
        workers = []
        for i in range(cores):
            p = mpl.Process(target=queue_worker, args=(q,))
            p.daemon = True
            p.start()
            workers.append(p)

        for i, f in enumerate(layers):
            # get all new activations
            with torch.no_grad():
                sec = [f(e) for e in first]

            # add to queue for workers
            q.put((self.inverse_sequentials[i], self.optimiers[i], tuple(zip(sec, first)), its))

            first = sec

        for i in range(cores):
            q.put("DONE")
        [w.join() for w in workers]
        # print(q.get())

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

    def train_parallel(self, train_data: List, its: int = 1, cores: int = mpl.cpu_count()):
        td = (zip(train_data[lvl + 1], train_data[lvl]) for lvl in range(len(conet)))
        with mpl.Pool(cores) as p:
            p.starmap(worker, zip(self.inverse_sequentials, self.optimiers, td, repeat(its)))

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
    import time

    net = Network([28, 28], 10, 10, (1.76, 0.05), convolutions=[nn.Conv2d(1, 2, (5, 5), padding="valid")])
    exit()

    # necessary for cuda multiprocessing
    mpl.set_start_method("spawn", force=True)

    dataloader.DATAPATH = "../"

    train_data, test_data = dataloader.get_train_data(100)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    depth = 100
    epochs = [0, 1, 10, 30, 50, 100, 200]
    vars = [0.1, 1.6, 3.2]

    for var_ in vars:
        print(f"START VAR = {var_} =======================================================================")
        # train network
        print("Start training forward net ----------------------------------------------------------")
        torch.cuda.empty_cache()
        print(torch.cuda.mem_get_info(0))
        net = Network(28 ** 2, 10, depth, (var_, 0.05), device=device, size="shrinking")
        net.training(train_data, test_data, its=1)
        # for epoch in epochs:
        #    net = Network(28 ** 2, 10, depth, (var_, 0.05), device=device)
        #    loss, accs = net.training(train_data, test_data, epoch)
        #    _, acc = net.eval(test_data)
        #    print(f"Accuracy of {acc} reached for EPOCH = {epoch}")

        # handle conet
        # print("Start training co net ----------------------------------------------------------------")
        conet = ContraNetwork(net, device=device)

        td = conet.get_activations_sorted(train_data)
        ts = []
        for i in range(len(conet)):
            td_ = zip(td[i + 1], td[i])
            t1 = time.time()
            conet.train_lvl(i, td_)
            ts.append(time.time() - t1)
            print(f"{ts[-1]}")
        print(np.mean(ts), np.std(ts))

        # print(torch.cuda.mem_get_info(0))
        # t1 = time.time()
        # actis = conet.train_parallel_sequential(train_data, stop=20)
        # print(f"Time for Sequential: {time.time() - t1} s")

        # t1 = time.time()
        # actis = conet.train_queue(train_data)
        # print(f"Time for Queue: {time.time() - t1} s")

        # t1 = time.time()
        # activations = conet.get_activations_sorted(train_data)
        # print(f"Time for data collection: {time.time() - t1} s")

        # t1 = time.time()
        # conet.train_parallel(activations)
        # print(f"Time for Conet-training: {time.time() - t1} s")

        # plot reconstructions
        for inp, _ in test_data:
            cascades = conet.cascade(inp)
            for cascade in cascades:
                plt.imshow(cascade[0].reshape((28, 28)))
                plt.show()
