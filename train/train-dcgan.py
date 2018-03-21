from tensorflow.python.keras.layers import Input, Conv2D, Conv2DTranspose, Dense, Reshape, BatchNormalization, Flatten
from tensorflow.python.keras.models import Model
import tensorflow as tf
import utils
import sys


def create_generator(input_shape):
    input_vec = Input(input_shape, name='input')
    x = Dense(128*4*4, activation='tanh', name='dense')(input_vec)
    x = BatchNormalization()(x)
    x = Reshape((4, 4, 128))(x)
    x = Conv2DTranspose(512, (3, 3), strides=2, activation=tf.nn.leaky_relu, padding='same', name='de_conv_1')(x)
    x = Conv2DTranspose(256, (3, 3), strides=2, activation=tf.nn.leaky_relu, padding='same', name='de_conv_2')(x)
    x = Conv2DTranspose(128, (3, 3), strides=2, activation=tf.nn.leaky_relu, padding='same', name='de_conv_3')(x)
    output = Conv2D(3, (3, 3), activation='tanh', padding='same', name='conv_2')(x)

    model = Model(input_vec, outputs=output)

    return model


def create_discriminator(input_shape=(32, 32, 3)):
    input_vec = Input(input_shape, name='input')
    x = BatchNormalization()(input_vec)
    x = Conv2D(64, (3, 3), strides=(2, 2), activation=tf.nn.leaky_relu, padding='same', name='d_conv_1')(x)
    x = Conv2D(128, (3, 3), strides=(2, 2), activation=tf.nn.leaky_relu, padding='same', name='d_conv_2')(x)
    x = Conv2D(256, (3, 3), strides=(2, 2), activation=tf.nn.leaky_relu, padding='same', name='d_conv_3')(x)
    x = Flatten()(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(input_vec, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adadelta')

    return model


if __name__ == "__main__":
    args = utils.parse_cli_args(sys.argv[1:], 'Train a dcgan to generate 32x32 images', support_directory_input=True)
    x_train_real = utils.load_images(args.data_dir)
    x_train_real = (x_train_real - 127.5) / 127.5

    utils.train(
        x_train_real, args.input_size, args.batch_size, args.epochs,
        args.output_dir, create_generator, create_discriminator)
