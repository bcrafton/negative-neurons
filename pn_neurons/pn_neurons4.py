

import numpy as np
import tensorflow as tf
import keras

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

assert(np.shape(x_train) == (50000, 32, 32, 3))
x_train = x_train / np.std(x_train, axis=0, keepdims=True)
x_train = np.concatenate((x_train, -1. * x_train), axis=3)
y_train = keras.utils.to_categorical(y_train, 10)

assert(np.shape(x_test) == (10000, 32, 32, 3))
x_test = x_test / np.std(x_test, axis=0, keepdims=True)
x_test = np.concatenate((x_test, -1. * x_test), axis=3)
y_test = keras.utils.to_categorical(y_test, 10)

####################################

f0 = 6
f1 = 64
f2 = 96
f3 = 128

####################################

epochs = 25
batch_size = 1
x = tf.placeholder(tf.float32, [1, 32 , 32 , f0])

####################################

weights = np.load('weights.npy').item()

w1 = tf.Variable(weights['w1'], dtype=tf.float32)
b1 = tf.Variable(np.zeros(shape=[1,32,32,128]), dtype=tf.float32)

w2 = tf.Variable(weights['w2'], dtype=tf.float32)
b2 = tf.Variable(np.zeros(shape=[1,16,16,192]), dtype=tf.float32)

w3 = tf.Variable(weights['w3'], dtype=tf.float32)
b3 = tf.Variable(np.zeros(shape=[1,8,8,256]), dtype=tf.float32)

####################################

def conv_op(x, w, b):
    convp = tf.nn.relu(tf.nn.conv2d(x, w[0], [1,1,1,1], 'SAME'))
    convn = tf.nn.relu(tf.nn.conv2d(x, w[1], [1,1,1,1], 'SAME'))
    conv = tf.concat((convp, -1. * convn), axis=3)
    return conv

####################################

conv1 = conv_op(x, w1, b1)
pool1 = tf.nn.avg_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

conv2 = conv_op(pool1, w2, b2)
pool2 = tf.nn.avg_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

conv3 = conv_op(pool2, w3, b3)
pool3 = tf.nn.avg_pool(conv3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

grad  = tf.gradients([conv3], [conv1])

####################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

xs = np.reshape(x_train[0], [1,32,32,6])

[c1, c2, c3, grad] = sess.run([conv1, conv2, conv3, grad], feed_dict={x: xs})

print (np.shape(grad))






























 
