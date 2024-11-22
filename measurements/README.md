# Measurements

Description of the different scripts in this folder

## Collect data
`collect_data.py` used to collect entropy and trainability data for feed forward networks. 
The data are stored in `.csv`-files. If you apply `imshow` to the cascades, you can 
visualize the reconstructions. Note that `cascade[0]` is a batch of input images.


## Collect falling
`collect_falling.py` collects trainability data for feedforward networks, starting a the crticial point
and maximum height, it trains networks and reduces the depth once their accuracy becomes to low.


##  Plot infos
`plot_infos.py` is used to plot the entropy collected by `collect_data.py` or similar.


## Plot falling
`plot_falling.py` is used to plot the results collected by `collect_falling.py`.


## train_single
`train_single.py` is an example script for training a single feedforward network and its 
corresponding conet.


## CNN reconstruction
`roi.py` is an example using VGG16 and the Imagenet dataset. It is used to reconstruct 
different layers and even different channels of the network. Please note that due to the heavy
shrinking in the first fully connected layer, this layer is harder to reconstruct.


## Resnet entropy
`collect_data_resnet.py` is used to collect data for a ResNet similar to `collect_data.py` for mpl.


## Convolution entropy
`collect_data_cnn.py` is used to collect data for a CNN similar to `collect_data.py` for mpl.
