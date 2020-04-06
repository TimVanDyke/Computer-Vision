import numpy as np
import theano
import theano.tensor as T
import gzip

import lasagne


# import matplotlib.pyplot as plt


def load_images(filename):

    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    # The inputs are vectors now, we reshape them to monochrome 2D images,
    # following the shape convention: (examples, channels, rows, columns)
    data = data.reshape(-1, 1, 28, 28)
    # The inputs come as bytes, we convert them to float32 in range [0,1].
    # (Actually to range [0, 255/256], for compatibility to the version
    # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
    return data / np.float32(256)


def load_labels(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    # The labels are vectors of integers now, that's exactly what we want.
    return data


train_labels = load_labels('./datasets/emnist-letters-train-labels-idx1-ubyte.gz')  # Y
train_images = load_images('./datasets/emnist-letters-train-images-idx3-ubyte.gz')  # X
test_labels = load_labels('./datasets/emnist-letters-test-labels-idx1-ubyte.gz')
test_images = load_images('./datasets/emnist-letters-test-images-idx3-ubyte.gz')






def build_mlp(input_var=None):
    # This creates an MLP of two hidden layers of 800 units each, followed by
    # a softmax output layer of 10 units. It applies 20% dropout to the input
    # data and 50% dropout to the hidden layers.

    # Input layer, specifying the expected input shape of the network
    # (unspecified batchsize, 1 channel, 28 rows and 28 columns) and
    # linking it to the given Theano variable `input_var`, if any:
    l_in = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                     input_var=input_var)

    # Apply 20% dropout to the input data:
    l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)

    # Add a fully-connected layer of 800 units, using the linear rectifier, and
    # initializing weights with Glorot's scheme (which is the default anyway):
    l_hid1 = lasagne.layers.DenseLayer(
        l_in_drop, num_units=800,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform())

    # We'll now add dropout of 50%:
    l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.5)

    # Another 800-unit layer:
    l_hid2 = lasagne.layers.DenseLayer(
        l_hid1_drop, num_units=800,
        nonlinearity=lasagne.nonlinearities.rectify)

    # 50% dropout again:
    l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.5)

    # Finally, we'll add the fully-connected output layer, of 10 softmax units:
    l_out = lasagne.layers.DenseLayer(
        l_hid2_drop, num_units=10,
        nonlinearity=lasagne.nonlinearities.softmax)

    # Each layer is linked to its incoming layer(s), so we only need to pass
    # the output layer to give access to a network in Lasagne:
    return l_out


input_var = T.tensor4('inputs')
target_var = T.ivector('targets')

network = build_mlp(input_var)

prediction = lasagne.layers.get_output(network)
loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
loss = loss.mean()
params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.nesterov_momentum(
    loss, params, learning_rate=0.01, momentum=0.9)

train_fn = theano.function([input_var, target_var], loss, updates=updates)


num_trainings = 100

for step in range(num_trainings):
    print('current step is ' + str(step))
    train_err = train_fn(train_images, train_labels)


test_predicion = lasagne.layers.get_output(network)
val_fn = theano.function([input_var], test_predicion)


# Optionally, you could now dump the network weights to a file like this:
np.savez('model.npz', *lasagne.layers.get_all_param_values(network))




for i in range(100):
    print(val_fn([test_images[i]]))

