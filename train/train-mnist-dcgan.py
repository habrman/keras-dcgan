from tensorflow.python.keras.layers import Input, Conv2D, Conv2DTranspose, Dense, Reshape, BatchNormalization, Flatten
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.datasets import mnist
import tensorflow as tf
import numpy as np
import utils
import sys


def create_generator(input_shape):
    input_vec = Input(input_shape, name='input')
    x = Dense(128*7*7, activation='tanh', name='dense')(input_vec)
    x = BatchNormalization()(x)
    x = Reshape((7, 7, 128))(x)
    x = Conv2DTranspose(128, (3, 3), strides=2, activation=tf.nn.leaky_relu, padding='same', name='de_conv_0')(x)
    x = Conv2DTranspose(64, (3, 3), strides=2, activation=tf.nn.leaky_relu, padding='same', name='de_conv_1')(x)
    output = Conv2D(1, (3, 3), activation='tanh', padding='same', name='output')(x)

    model = Model(input_vec, outputs=output)

    return model


def create_discriminator(input_shape=(28, 28, 1)):
    input_vec = Input(input_shape, name='input')
    x = BatchNormalization()(input_vec)
    x = Conv2D(16, (3, 3), strides=(2, 2), activation=tf.nn.leaky_relu, padding='same', name='conv_0')(x)
    x = Conv2D(32, (3, 3), strides=(2, 2), activation=tf.nn.leaky_relu, padding='same', name='conv_1')(x)
    x = Conv2D(64, (3, 3), strides=(2, 2), activation=tf.nn.leaky_relu, padding='same', name='conv_2')(x)
    x = Flatten()(x)
    output = Dense(1, activation='sigmoid', name='output')(x)

    model = Model(input_vec, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adadelta')

    return model


if __name__ == "__main__":
    args = utils.parse_cli_args(sys.argv[1:], 'Train a dcgan to generate 28x28 mnist style images')

    (x_train_real, _), _ = mnist.load_data()
    x_train_real = np.expand_dims((x_train_real - 127.5) / 127.5, axis=-1)

    utils.train(
        x_train_real, args.input_size, args.batch_size, args.epochs,
        args.output_dir, create_generator, create_discriminator)
