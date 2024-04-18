# Measurements using differential entropy

In this folder, we present the scrips for running measurements on the MNIST and CIFAR10 data in our paper.
The data are taken using torchvision.

Please note that the parameters (especially training Epochs) is reduced in the script to allow for exemplary runs.
Training a network 10 Epochs is NOT sufficient for determining trainability and the plots thus will differ to the ones
in the paper. In case you want to recreate these plots please control that the same parameters are used.

## Scripts

- train_single.py: Shows a basic example of how to train a neural network and apply our method
(in this case after training to create nice images using the reconstructions)

- collect_data.py: File used to collect the cutoff-images presented in the paper (use plot_data.py to show)

- plot_data.py: Used to show collected data from collect_data.py

- collect_falling.py: File used to collect trainability data with a minimal accuracy. If the accuracy is not
archived, a more shallow network is considered and trained.