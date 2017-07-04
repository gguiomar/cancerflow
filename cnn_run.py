import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math

# my bibs
from cnn_net_bib import * 
from cnn_plot_bib import *

# Convolutional Layer 1.
filter_size1 = 2          # Convolution filters are 5 x 5 pixels.
num_filters1 = 10         # There are 16 of these filters.

# Convolutional Layer 2.
filter_size2 = 2          # Convolution filters are 5 x 5 pixels.
num_filters2 = 20         # There are 36 of these filters.

# Fully-connected layer.
fc_size = 80             # Number of neurons in fully-connected layer.

# Loading the MNIST data
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST/', one_hot=True)

data.test.cls = np.argmax(data.test.labels, axis=1)

# MNIST data dimensionality

img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)
num_channels = 1 # grey scale
num_classes = 10 


# placeholder variables 

x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

# network architecture

layer_conv1, weights_conv1 = new_conv_layer(input=x_image,
                   num_input_channels=num_channels,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   use_pooling=True)

layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1,
                   num_input_channels=num_filters1,
                   filter_size=filter_size2,
                   num_filters=num_filters2,
                   use_pooling=True)

layer_flat, num_features = flatten_layer(layer_conv2)

layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size,
                         use_relu=True)

layer_fc2 = new_fc_layer(input=layer_fc1,
                         num_inputs=fc_size,
                         num_outputs=num_classes,
                         use_relu=False)



# network predictions

y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, dimension=1)

# cost function and optimization measures

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

# performance measures

correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# running the network

session = tf.Session()
session.run(tf.global_variables_initializer())
train_batch_size = 64

# Counter for total number of iterations performed so far.

# optimization algorithm
total_iterations = 0
def optimize(num_iterations):

	# Ensure we update the global variable rather than a local copy.
	global total_iterations

	# Start-time used for printing time-usage below.
	start_time = time.time()

	for i in range(total_iterations,
	               total_iterations + num_iterations):

	    # Get a batch of training examples.
	    # x_batch now holds a batch of images and
	    # y_true_batch are the true labels for those images.
	    x_batch, y_true_batch = data.train.next_batch(train_batch_size)

	    # Put the batch into a dict with the proper names
	    # for placeholder variables in the TensorFlow graph.
	    feed_dict_train = {x: x_batch,
	                       y_true: y_true_batch}

	    # Run the optimizer using this batch of training data.
	    # TensorFlow assigns the variables in feed_dict_train
	    # to the placeholder variables and then runs the optimizer.
	    session.run(optimizer, feed_dict=feed_dict_train)

	    # Print status every 100 iterations.
	    if i % 100 == 0:
	        # Calculate the accuracy on the training-set.
	        acc = session.run(accuracy, feed_dict=feed_dict_train)

	        # Message for printing.
	        msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"

	        # Print it.
	        print(msg.format(i + 1, acc))

	# Update the total number of iterations performed.
	total_iterations += num_iterations

	# Ending time.
	end_time = time.time()

	# Difference between start and end-times.
	time_dif = end_time - start_time

	# Print the time-usage.
	print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

optimize(num_iterations = 5000)
