# Measurements using differential entropy

In this folder, we present the scrips for running measurements on the MNIST and CIFAR10 data in our paper.
The data are taken using torchvision.

## Scripts

- train_single.py: Shows a basic example of how to train a neural network and apply our method
(in this case after training to create nice images using the reconstructions)

- collect_data.py: File used to collect the cutoff-images presented in the paper

- falling.py: File used to collect trainability and cutoff-images with a minimal accuracy. If the accuracy is not
archived, a more shallow network is considered and trained.