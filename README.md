# cnn-mwa
This application is an image classification tool that aims at classifying flux density images observed by the MWA telescope. The classes specify the suitability of the images' incorporation into a sky survey. It is based on a convolutional neural network and uses the PyTorch framework. 

This repository provides the necessary code to train a model on fits image files: *scripts/cnn_gleam_train.py*. It also provides a pre-trained model *GleamNet* that can be directly used as the classifier with the code from *scripts/cnn_gleam_test.py*. 

The network's initial aim was to identify images from the GLEAM and GLEAM-X dataset that showed artifacts from either RFI or from undeconvolved sources in the sidelobes of the telescope's primary beam (cf. Rettelbach & Hurley-Walker 2019, in prep). The model's architecture is therefore optimized for solving this specific problem. To use this architecture for solving other problems via transfer learning, certain specifications and hyperparameters will need to be adapted. 

For each image, the classifier will output probabilities for each target class. This results then will be added to the respective image's header file. If desired, the code will also output statistics for the classification, such as losses and the certainty of a prediction.

## Installation:
In order to run and make use of all features, you will need to install the following libraries:
- ```pytorch```
- ```astropy```
- ```glob2```
- ```pandas```
- ```PIL```
- ```scikit-learn```

The repo provides a Dockerfile with which a Docker-image was created. It includes all necessary libraries.
A container from the image can be built with:

```docker pull trettelbach/pytorch_astro```

For guides on installing Docker, please refer to the [official Docker documentation](https://docs.docker.com/)



## Usage:
The repository contains three scripts which make use of the neural network. All scripts are coded with python and are conceived as command-line applications.
### Testing/Validating/Using only
The script ```scripts/cnn_gleam_use.py``` loads the saved weights of the pre-trained network *GleamNet* and runs the model on a validation/testing/unknown dataset. The fits header file will be updated to contain information on the classifications made by the network. If desired, it will also output statistics on: #TODO

It can be run with the following:

```docker run trettelbach/pytorch_astro:latest python <PATH TO cnn_gleam_use.py> <PATH TO THE DIRECTORY FOR OUTPUTS> <STATS>```

```<STATS>``` determines if further statistics of the run (such as certainties, weights, feature maps) shall be output as well. It can either assume the values 0 (no statistics) or 1 (statistics).

When running the application on HPC systems, such as clusters by the Pawsey Supercomputing Centre, *Shifter* will need to be used instead of Docker. 

### Training only
The script ```scripts/cnn_gleam_train.py``` trains a neural model with the *CnnGleam* architecture and saves the model weights. If desired, it will also output statistics. If changes to hyperparameters (such as the number of epochs for training) need to be made, this must be adapted within the script.

It can be run with the following:

```docker run trettelbach/pytorch_astro:latest python <PATH TO cnn_gleam_train.py> <PATH TO THE DIRECTORY FOR OUTPUTS> <STATS>```

### Training and testing
If you wish to re-train the network and directly test it with a control dataset, the script ```scripts/cnn_gleam_all.py``` provides this option. If desired, it will also output statistics.

It can be run with the following:

```docker run trettelbach/pytorch_astro:latest python <PATH TO cnn_gleam_all.py> <PATH TO THE DIRECTORY FOR OUTPUTS> <STATS>```


# Credits:

# License
The code is published under the xxx license.
