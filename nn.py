
import lasagne

def build_mlp(input_var=None):
    # Here we create a MLP and in our case we will be adding 2 hidden layers with
    # 800 nodes in each layer. The reason for this is because according to 
    # http://yann.lecun.com/exdb/mnist/ they have one of the best results.

    # The input will be a dataset in the form of a numpy array with this shape.
    # Each picture in the data set is 28x28 px and has only one color layer
    l_in = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                     input_var=input_var)

    # Apply 10% dropout to the input data:
    l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.1)

    # Add a fully-connected layer of 800 units
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

    # Finally, we'll add the fully-connected output layer, of 47 units
    # because our dataset has 47 different characters.
    # These can be found in the dataset.py file.
    l_out = lasagne.layers.DenseLayer(
        l_hid2_drop, num_units=47,
        nonlinearity=lasagne.nonlinearities.softmax)

    # Each layer is linked to its incoming layer(s), so we only need to pass
    # the output layer to give access to a network in Lasagne:
    return l_out