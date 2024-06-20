# Infoflow-paper

Official repository to XXXXX.

### Please check the paper for more details and better description

Abstract:
An important challenge in machine learning is to predict the initial conditions under which a given neural network will
be trainable. We present a method for predicting the trainable regime in parameter space for deep feedforward neural
networks, based on reconstructing the input from subsequent activation layers via a cascade of single-layer auxiliary
networks. For both MNIST and CIFAR10, we show that a single epoch of training of the shallow cascade networks is
sufficient to predict the trainability of the deep feedforward network, thereby providing a significant reduction in
overall training time. We achieve this by computing the relative entropy between reconstructed images and the original
inputs, and show that this probe of information loss is sensitive to the phase behaviour of the network. Our results
provide a concrete link between the flow of information and the trainability of deep neural networks, further
elucidating the role of criticality in these systems.

## General concept

The idea of this repository and the article is to measure the entropy in deep neural networks to predict trainability
prior to training. This is done by using a set of reconstruction networks we call cascade. These cascades do not only
allow for quantifiying the information in the hidden layers of neural networks (and subsequently identify regions of
best training) but also allow for a first step in interpreting and understanding the decision making process.
The hole procedure of how the reconstructions work is illustrated in the following figure.
![Sketch of the procedure using cascades](images/scheme.png)

## Measuring the entropy

As shown in [Schoenholz](https://arxiv.org/abs/1611.01232) neural networks experience a phase transition in the variance
of weights during initialization. Thereby, training is optimal close to the phase transition (critical regime). Using
the differential or relative entropy, these regimes can be identified prior to training.

This can be seen in the following measurement of differential entropy for an untrained deep neural network on the MNIST
data. The different colors encode the differential entropy. At a variance of around 1.7 a blue cone can be seen. At
these variances, the neural network is able to propagate the information into deeper layers (bigger differential
entropy)
allowing for better trainability. In the ordered regime (left side, red) the information are lost fast indicating a
not-trainable regime. In the chaotic phase (right side, green/yellow-ish) some information are still propagated into
deeper layers of the neural network. Therefore, the networks are more trainable but less good as in the critical phase.
This matches with the results in [Schoenholz](https://arxiv.org/abs/1611.01232), where the trainability is determined
experimentally by training the networks.
![Image of differential entropy](images/diff_entropy.png)

## Interpreting the networks decisions

Applying the cascades, we can observe the change in features the network performs over its depth. An example is given in
the following image.

In the first line, the reconstructions of an input 5 are shown for the 1st to L-th (L = 9) layer. As can be seen, the
network seems to sharpen some features, in particular, the upper left edge of the 5, connecting its main-body and
horizontal bar. This is rather interesting as for the next input
(also with label 5) this is the exact features which is not present and the network connects both with a smooth line.
This might cause the network to wrongly predict the result as a 6 rather than a 5.

At last, by appling the method on artifical activation vectors, we can construct an input the network would consider
optimal for this class. This is shown in the lowest part of the image, where different number are created through the
reconstruction.

![Image of example reconstructions](images/reconstruction.png)
