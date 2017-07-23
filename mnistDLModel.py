import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('data', one_hot=True)
type(mnist)
mnist.train.images.shape
sample = mnist.train.images[2].reshape(28, 28)

plt.imshow(sample, cmap='Greys')

learning_rate = 0.001
training_epochs = 15
batch_size = 100

n_classes = 10
n_samples = mnist.train.num_examples

n_input = 784

n_hidden_1 = 256
n_hidden_2 = 256


def multilayer_perceptron(x, weight, biases):
    '''
    :param x: placeholder for data input
    :param weight: a dictionary of weights
    :param biases: a dictionary of bias values
    :return:
    '''


# First hidden layer with RELU activation
# X*W+B
layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])

# RELU(X*W+b) => f(x) = max(0,x)
layer_1 = tf.nn.relu(layer_1)

# Second hidden layer with RELU activation
# layer_1*W+B
layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])

# RELU(layer_1*W+b) => f(x) = max(0,layer_1)
layer_2 = tf.nn.relu(layer_2)

# Last output layer
out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
return out_layer

weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

x = tf.placeholder('float', [None, n_input])
y = tf.placeholder('float', [None, n_classes])

pred = multilayer_perceptron(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

