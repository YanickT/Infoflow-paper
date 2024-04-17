# Differential entropy

This subpackage holds the scripts for calculating the 
cutoff using [differential entropy](https://de.wikipedia.org/wiki/Differentielle_Entropie).

The differential entropy is calculated pixel-wise, where for the activation
of a single pixel for different inputs a Gaussian distribution (when information are lost)
are assumed. For a Gaussian distribution, the differential entropy is calculated as 
described [here](https://math.stackexchange.com/questions/1804805/how-is-the-entropy-of-the-normal-distribution-derived)


## Note
The Contra-Network uses a hijack-method for finding the sequence of layers in the primary network.
It searches for a torch.nn.Sequential object in the Networks class. 
If this is not found, the Contra-Network is not able to create its requiered structure.
If you plan to use this scripts, implement your network accordingly or modify this script.
In case of a modification of the script, please send a merge request such that this code might be able to handle
most cases over time.
