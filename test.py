import numpy as np
import theano
import theano.tensor as T
from nn import build_mlp
from load import load_data
import sys

import lasagne

train_images, train_labels, test_images, test_labels = load_data(sys.argv[1])

input_var = T.tensor4('inputs')
target_var = T.ivector('targets')

network = build_mlp(input_var)

#load network
with np.load('./networks/{}.npz'.format(sys.argv[2])) as f:
    param_values = [f['arr_%d' % i] for i in range(len(f.files))]
lasagne.layers.set_all_param_values(network, param_values)


# run 100 examples
val_fn = theano.function(
    [input_var], lasagne.layers.get_output(network, deterministic=True))

correct = 0
num_tests = 100

for i in range(num_tests):
    output = val_fn([test_images[i]])
    max = np.argmax(output[0])
    if max == test_labels[i]:
        correct += 1
    print('Predicted : {}, actual: {}'.format(max, test_labels[i]))

acc_for_tests = correct/num_tests * 100

print("Accuracy for the first {} charactes is : {}".format(num_tests, acc_for_tests))


# run agains the whole dataset
test_prediction = lasagne.layers.get_output(network)
test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1),
                       target_var), dtype=theano.config.floatX)

acc_fn = theano.function([input_var, target_var], test_acc)

print("Accuracy for the entire test dataset is: {}%".format(
    acc_fn(test_images, test_labels)*100))

