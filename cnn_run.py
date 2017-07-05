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
from cnn_data_import import *

CURRENT_FOLDER = '/Users/indp/Dropbox/rotations/polavieja/code/'

folder = CURRENT_FOLDER + 'data/good_datasets/isic-archive_ss/'

path_melanoma = folder+ '/sq_mel/'
path_rotated_melanoma = folder + '/rot_mel/'
path_non_melanoma = folder + '/sq_nmel'

img_path2 = folder + '/all/'
img_path = folder + '/all2/'

data = read_data_sets(train_dir = folder, num_classes = 2)

a = load_cancer_data_labels(path_melanoma, path_non_melanoma, path_rotated_melanoma)[:,1].astype(int)
print(sum(a == 1), sum(a == 0))

img_name_cls = load_cancer_data_labels(path_melanoma, path_non_melanoma, path_rotated_melanoma)

img_shape = (400, 400)
#img_shape = get_expected_size(img_name_cls, img_path)
img_size_flat = img_shape[0] * img_shape[1]
num_channels = 1 # grey scale
num_classes = 2


## NETWORK DEFINITION

# Convolutional Layer 1.
filter_size1 = 6          # Convolution filters are 5 x 5 pixels.
num_filters1 = 16         # There are 16 of these filters.

# Convolutional Layer 2.
filter_size2 = 6          
num_filters2 = 32        

filter_size3 = 6          
num_filters3 = 64         

filter_size4 = 6          
num_filters4 = 128       

# Fully-connected layer.
fc_size = 128             


# placeholder variables 

x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_shape[0], img_shape[1], num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)
weight = tf.placeholder(tf.float32, shape=(), name='weight')

# Network architecture - Tensorflow definitions

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

layer_conv3, weights_conv3 = new_conv_layer(input=layer_conv2,
                   num_input_channels=num_filters2,
                   filter_size=filter_size3,
                   num_filters=num_filters3,
                   use_pooling=True)

layer_conv4, weights_conv4 = new_conv_layer(input=layer_conv3,
                   num_input_channels=num_filters3,
                   filter_size=filter_size4,
                   num_filters=num_filters4,
                   use_pooling=True)

layer_flat, num_features = flatten_layer(layer_conv4)

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

# cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)
# using weighted cross entropy due to dataset inbalance

cross_entropy = tf.nn.weighted_cross_entropy_with_logits(targets = y_true, 
                                                         logits = layer_fc2, 
                                                         pos_weight = weight)

cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

# performance measures

correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# running the network

session = tf.Session()
session.run(tf.global_variables_initializer())
session.run(tf.local_variables_initializer())
train_batch_size = 50

# optimization algorithm
total_iterations = 0

acc = []
cost = []
tpv = []
fpv = []

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
        x_batch, y_true_batch, weight_batch = data.train.next_batch(train_batch_size)
        
        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch,
                           weight: weight_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict = feed_dict_train)
        
        # non_optimal way of calculating recall and precision
        yt = y_true_batch
        yt_cls = np.asarray(tf.argmax(yt, dimension = 1).eval(feed_dict={x: x_batch}, session=session))
        
        yp = y_pred.eval(feed_dict={x: x_batch}, session=session)
        yp_cls = np.asarray(tf.argmax(yp, dimension = 1).eval(feed_dict={x: x_batch}, session=session))
        
        tp = np.sum(np.asarray([1 if e == d and e == 1 else 0 for e,d in zip(yt_cls, yp_cls)]))
        tn = np.sum(np.asarray([1 if e == d and e == 0 else 0 for e,d in zip(yt_cls, yp_cls)]))
        fp = np.sum(np.asarray([1 if e != d and e == 0 else 0 for e,d in zip(yt_cls, yp_cls)]))
        fn = np.sum(np.asarray([1 if e != d and e == 1 else 0 for e,d in zip(yt_cls, yp_cls)]))
        
        print('TP:', tp, 'TN:', tn, 'FP:', fp, 'FN:', fn)
        print('Recall: ', tp/float(tp+fn), 'Precision: ', tp/float(tp+fp))
        
        tpv.append(tp)
        fpv.append(fp)
        
        # Print status every n iterations.
        if i % 1 == 0:
            # Calculate the accuracy on the training-set.
            acc.append(session.run(accuracy ,feed_dict=feed_dict_train))
            cost.append(session.run(cost, feed_dict=feed_dict_train))

            # Message for printing.
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"

            # Print it.
            print(msg.format(i + 1, acc[i]))
            # print('Recall: ', rec, 'Precision: ', prec) # not usefull
    
    # Update the total number of iterations performed.
    total_iterations += num_iterations

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

# RUN THE CODE
optimize(num_iterations = 2)