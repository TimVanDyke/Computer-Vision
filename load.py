import numpy as np
import gzip


def load_data(dataset):
    def load_images(filename):

        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        data = data.reshape(-1, 1, 28, 28)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        return data / np.float32(256)

    def load_labels(filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are a list of integers integers
        return data

    train_labels = load_labels(
        './datasets/emnist-{}-train-labels-idx1-ubyte.gz'.format(dataset))  # Y
    train_images = load_images(
        './datasets/emnist-{}-train-images-idx3-ubyte.gz'.format(dataset))  # X
    test_labels = load_labels(
        './datasets/emnist-{}-test-labels-idx1-ubyte.gz'.format(dataset))   # Y
    test_images = load_images(
        './datasets/emnist-{}-test-images-idx3-ubyte.gz'.format(dataset))  # X

    return train_images, train_labels, test_images, test_labels
