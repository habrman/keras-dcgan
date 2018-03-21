from tensorflow.python.keras.layers import Conv2D, Input, Conv2DTranspose, Reshape, BatchNormalization, Flatten, Dense
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.callbacks import TensorBoard
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import argparse
import imageio
import glob
import math
import os


# Custom loss for generator to avoid vanishing gradients when training generator
def custom_loss():
    def loss(y_true, y_pred):
        return -tf.log(y_pred)
    return loss


def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()


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


def create_gan(input_size):
    generator = create_generator((input_size,))
    discriminator = create_discriminator()

    for layer in discriminator.layers:
        layer.trainable = False

    input_vec = Input((input_size,), name='input')
    output = discriminator(generator(input_vec))

    gan = Model(input_vec, outputs=output)

    gan.compile(loss=custom_loss(), optimizer='adadelta')
    return gan, discriminator, generator


def create_noise_batch(samples, size):
    return np.random.normal(0, 0.2, size=(samples, size))


def save_predictions(generator, epoch, step, input_size, pred_dir):
    img_size = 28
    sqrt_samples = 10

    samples = create_noise_batch(sqrt_samples**2, input_size)
    pred = generator.predict_on_batch(samples)

    img = np.zeros((img_size*sqrt_samples, img_size*sqrt_samples, 1), dtype=np.uint8)
    for i, p in enumerate(pred):
        img[
            (i // sqrt_samples)*img_size: ((i // sqrt_samples)+1)*img_size,
            (i % sqrt_samples)*img_size: ((i % sqrt_samples) + 1)*img_size, :] = np.array((p * 127.5) + 127.5, dtype=np.uint8)

    imageio.imwrite(os.path.join(pred_dir, 'pred%d-%d.png' % (epoch, step)), img)
    return


def train(input_size, batch_size, epochs, output_dir):
    pred_dir = os.path.join(output_dir, 'predictions')
    model_dir = os.path.join(output_dir, 'models')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    gan, discriminator, generator = create_gan(input_size)
    (x_train_real, _), _ = mnist.load_data()

    # Normalize mnist data between -1 and 1
    x_train_real = np.expand_dims((x_train_real - 127.5) / 127.5, axis=-1)

    batch_size = math.ceil(batch_size / 2)
    y_batch_ones_generator = np.ones([batch_size*2, 1])
    y_batch_zeros = np.zeros([batch_size, 1])

    callback = TensorBoard(log_dir=os.path.join(output_dir, 'log'))
    callback.set_model(gan)

    indices = np.arange(x_train_real.shape[0])
    np.random.shuffle(indices)
    batch_indices = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]

    step = 0
    for e in range(epochs):
        print('Epoch: %d/%d' % (e + 1, epochs))

        for indices in tqdm(batch_indices):
            # Create batch data
            x_batch_real = x_train_real[indices]
            noise_batch = create_noise_batch(len(indices), input_size)
            x_batch_generator = generator.predict_on_batch(noise_batch)

            # Train discriminator with label smoothing
            y_batch_ones = np.random.uniform(0.8, 1, size=(len(indices), 1))

            discriminator_loss_m = discriminator.train_on_batch(x_batch_real, y_batch_ones)
            discriminator_loss_g = discriminator.train_on_batch(x_batch_generator, y_batch_zeros[:len(indices)])
            discriminator_loss = (discriminator_loss_m + discriminator_loss_g) / 2

            # Train generator
            for _ in range(2):
                noise_batch = create_noise_batch(len(indices) * 2, input_size)
                generator_loss = gan.train_on_batch(noise_batch, y_batch_ones_generator[:2 * len(indices)])
            write_log(
                callback, ['discriminator_train_loss', 'generator_train_loss'],
                [discriminator_loss, generator_loss], step)
            step += 1
            if step % 100 == 0:
                save_predictions(generator, e, step, input_size, pred_dir)

        discriminator.save(os.path.join(model_dir, 'discriminator_%d.h5' % e))
        generator.save(os.path.join(model_dir, 'generator_%d.h5' % e))

        save_predictions(generator, e, step, input_size, pred_dir)
    generate_gif(pred_dir, output_dir)


def generate_gif(pred_dir, output_dir):
    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        import re
        return [atoi(c) for c in re.split('(\d+)', text)]

    images = []
    img_paths = glob.glob(pred_dir + '/*.png')
    img_paths.sort(key=natural_keys)
    for img_path in img_paths[::9]:
        images.append(imageio.imread(img_path))
    imageio.mimsave(os.path.join(output_dir, 'mnist_training.gif'), images, duration=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train mnist dcgan')
    parser.add_argument(
        '-i', '--input-size', type=int, default=100,
        help='Input vector size to the generator')
    parser.add_argument(
        '-b', '--batch-size', type=int, default=128,
        help='Batch size')
    parser.add_argument(
        '-e', '--epochs', type=int, default=100,
        help='Number of epochs to train')
    parser.add_argument(
        '-o', '--output-dir', default='./output',
        help='Output directory for logs, predictions and models')

    args = parser.parse_args()
    print(args)

    train(args.input_size, args.batch_size, args.epochs, args.output_dir)
