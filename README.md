# Keras dcgan
Train a deep convolutional generative adversarial network to generate mnist like data.

## Dependencies
* tensorflow
* tqdm
* imageio

## How to
There's two training scripts in the folder `train`.
1) `train-mnist-dcgan.py`
    Train a gan to generate mnist like images. It's provided as a simple example where the user doesn't need to provide the data set. It imports the mnist data set using keras. Run it with `python3 train/train-mnist-dcgan.py`.
2) `train-dcgan.py`
    Train a gan using your own data set. The size of the generated images is 32x32 so make sure your data set does not contain small details since the real data will be resized to 32x32 when training the discriminator. Run it with `python3 train/train-dcgan.py -d <your data folder>` where `<your data folder>` should contain your image data set in `.png` or `.jpg` format.

Both scripts create an output folder containing models for each epoch,  predictions for each 100 step and tensorboard logs with generator and discriminator loss for each training step. Both of them also provide input parameters for input size, batch size, number of training epochs and output directory that can be used.

## Network structure
![alt text](https://github.com/habrman/keras-dcgan/blob/master/assets/network-structure.png)

## MNIST training evolution
![alt text](https://github.com/habrman/keras-dcgan/blob/master/assets/mnist_training_example.gif)

## Fashion MNIST training evolution
![alt text](https://github.com/habrman/keras-dcgan/blob/master/assets/fashion_mnist_training_example.gif)
