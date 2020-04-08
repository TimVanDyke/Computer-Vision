import numpy as np
import theano
import theano.tensor as T
from nn import build_mlp
from load import load_data
from dataset import get_char

import lasagne


def letters_from_imgArr(dataset, netw, pictureArr):
    input_var = T.tensor4('inputs')

    # Build network schematic
    network = build_mlp(input_var)

    # Load trained network
    with np.load('./networks/{}.npz'.format(netw)) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)

   # This function uses the network to determine the character
    val_fn = theano.function(
        [input_var], lasagne.layers.get_output(network, deterministic=True))

    letters = []
    # Iterate through picutres in the arry and use the network to find the letter
    for i in range(len(pictureArr)):
        output = val_fn([pictureArr[i]])
        max = np.argmax(output[0])
        letters.append(get_char(max))
    return letters    

