# Entropy measurements

This folder contains the core-functionality for measuring the entropy in neural networks.
For measuring the entropy, two different methods are available. 
Both methods are presented in the following:

## Relative entropy

This subpackage holds the scripts for calculating the
cutoff using [relative entropy](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence).

The relative entropy is calculated between the input and the considered layers, where the activations are normalized
and interpreted as probability distributions.


## Differential entropy

This subpackage holds the scripts for calculating the 
cutoff using [differential entropy](https://de.wikipedia.org/wiki/Differentielle_Entropie).

The differential entropy is calculated pixel-wise, where for the activation
of a single pixel for different inputs a Gaussian distribution (when information are lost)
are assumed. For a Gaussian distribution, the differential entropy is calculated as 
described [here](https://math.stackexchange.com/questions/1804805/how-is-the-entropy-of-the-normal-distribution-derived)


## Notes
The Contra-Network in network.py uses a hijack-method for finding the sequence of layers in the primary network.
It searches for a torch.nn.Sequential object in the Networks class. 
If this is not found, the Contra-Network is not able to create its required structure.
If you plan to use this scripts, implement your network accordingly or modify this script.
In case of a modification of the script, please send a merge request such that this code might be able to handle
most cases over time.



