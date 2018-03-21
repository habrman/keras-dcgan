from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.callbacks import TensorBoard
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import argparse
import imageio
import glob
import math
import scipy
import os


def parse_cli_args(argv, description, support_directory_input=False):
    parser = argparse.ArgumentParser(description=description)

    if support_directory_input:
        parser.add_argument(
            '-d', '--data-dir', required=True,
            help='Input directory with .png images (real data)')

    parser.add_argument(
        '-i', '--input-size', type=int, default=100,
        help='Input vector size to the generator')
    parser.add_argument(
        '-b', '--batch-size', type=int, default=128,
        help='Batch size')
    parser.add_argument(
        '-e', '--epochs', type=int, default=20,
        help='Number of epochs to train')
    parser.add_argument(
        '-o', '--output-dir', default='./output',
        help='Output directory for logs, predictions and models')

    args = parser.parse_args(argv)
    print(args)
    return args


def generate_gif(pred_dir, output_dir):
    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        import re
        return [atoi(c) for c in re.split('(\d+)', text)]

    images = []
    img_paths = glob.glob(pred_dir + '/*.png')
    img_paths.sort(key=natural_keys)
    sample_interval = math.ceil(len(img_paths) / 10)
    for img_path in img_paths[::sample_interval]:
        images.append(imageio.imread(img_path))
    imageio.mimsave(os.path.join(output_dir, 'training.gif'), images, duration=1)


def save_predictions(generator, epoch, step, input_size, pred_dir):
    sqrt_samples = 10

    samples = create_noise_batch(sqrt_samples**2, input_size)
    pred = generator.predict_on_batch(samples)

    img_size = pred.shape[1]
    img = np.zeros((img_size*sqrt_samples, img_size*sqrt_samples, 3), dtype=np.uint8)
    for i, p in enumerate(pred):
        img[
            (i // sqrt_samples)*img_size: ((i // sqrt_samples)+1)*img_size,
            (i % sqrt_samples)*img_size: ((i % sqrt_samples) + 1)*img_size, :] = np.array((p * 127.5) + 127.5, dtype=np.uint8)

    imageio.imwrite(os.path.join(pred_dir, 'pred%d-%d.png' % (epoch, step)), img)
    return


def load_images(folder, shape=(32, 32, 3)):
    print('Loading real data')
    img_paths = []
    for ext in ('*.png', '*.jpg'):
        img_paths.extend([img_path for img_path in glob.iglob(folder + '/**/' + ext, recursive=True)])
    print(len(img_paths))

    if len(img_paths) == 0:
        raise ValueError('Could not find any .png images in data folder')

    images = [scipy.ndimage.imread(img_path, mode='RGB') for img_path in img_paths]

    if images[0].shape != shape:
        images = [scipy.misc.imresize(img, shape) for img in images]

    return np.stack(images)


def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()


def create_noise_batch(samples, size):
    return np.random.normal(0, 0.2, size=(samples, size))


# Custom loss for generator to avoid vanishing gradients when training generator
def generator_loss():
    def loss(y_true, y_pred):
        return -tf.log(y_pred)
    return loss


def create_gan(input_size, create_generator, create_discriminator):
    generator = create_generator((input_size,))
    discriminator = create_discriminator()

    for layer in discriminator.layers:
        layer.trainable = False

    input_vec = Input((input_size,), name='input')
    output = discriminator(generator(input_vec))

    gan = Model(input_vec, outputs=output)

    gan.compile(loss=generator_loss(), optimizer='adadelta')
    return gan, discriminator, generator


def train(x_train_real, input_size, batch_size, epochs, output_dir, create_generator, create_discriminator):
    pred_dir = os.path.join(output_dir, 'predictions')
    model_dir = os.path.join(output_dir, 'models')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    gan, discriminator, generator = create_gan(input_size, create_generator, create_discriminator)

    y_batch_ones_generator = np.ones([batch_size, 1])
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
                noise_batch = create_noise_batch(len(indices), input_size)
                generator_loss = gan.train_on_batch(noise_batch, y_batch_ones_generator[:len(indices)])
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
