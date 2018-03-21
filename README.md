# Keras dcgan
Train a deep convolutional generative adversarial network to generate mnist like data.

## Dependencies
* tensorflow
* tqdm
* imageio

## How to
Clone the repo and run `python3 mnist/train-mnist-gan.py`. This will create the output folder containing models and predictions for each epoch and tensorboard logs with generator and discriminator loss for each training step.

There are input parameters for input size, batch size, number of training epochs and output directory that can be used.

## Network structure
![alt text](https://github.com/habrman/keras-dcgan/blob/master/assets/network-structure.png)

## MNIST training evolution
![alt text](https://github.com/habrman/keras-dcgan/blob/master/assets/mnist_training_example.gif)
