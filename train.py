import numpy as np
import theano
import theano.tensor as T
import sys
from nn import build_mlp
from load import load_data

import lasagne

print('\nLoading Datasets\n')

train_images, train_labels, test_images, test_labels = load_data(sys.argv[1])

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

print('Datasets Loaded, training is starting\n')

num_trainings = int(sys.argv[3])

for step in range(num_trainings):
    print('current step is ' + str(step))
    train_err = train_fn(train_images, train_labels)


# save the network
np.savez('./networks/{}.npz'.format(sys.argv[2]),
         *lasagne.layers.get_all_param_values(network))

