

import numpy as np
import tensorflow as tf
import keras
from tensorflow.python.ops import gen_nn_ops

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

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

####################################

f0 = 6
f1 = 64
f2 = 96
f3 = 128

####################################

epochs = 25
batch_size = 1
x = tf.placeholder(tf.float32, [1, 32 , 32 , f0])
y = tf.placeholder(tf.float32, [1, 10])

####################################

weights = np.load('weights.npy').item()

w1 = tf.Variable(weights['w1'], dtype=tf.float32)
w2 = tf.Variable(weights['w2'], dtype=tf.float32)
w3 = tf.Variable(weights['w3'], dtype=tf.float32)

w4 = tf.Variable(weights['w4'], dtype=tf.float32)

####################################

def conv_op(x, w):
    convp = tf.nn.relu(tf.nn.conv2d(x, w[0], [1,1,1,1], 'SAME'))
    convn = tf.nn.relu(tf.nn.conv2d(x, w[1], [1,1,1,1], 'SAME'))
    conv = tf.concat((convp, -1. * convn), axis=3)
    return conv

####################################

conv1 = conv_op(x, w1)
pool1 = tf.nn.avg_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

conv2 = conv_op(pool1, w2)
pool2 = tf.nn.avg_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

conv3 = conv_op(pool2, w3)
pool3 = tf.nn.avg_pool(conv3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

flat = tf.reshape(pool3, [batch_size, 4*4*2*f3])
fc1 = tf.matmul(flat, w4)

loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=fc1)
[grad] = tf.gradients(loss, [conv1])

####################################

do     = tf.nn.softmax(fc1) - y
dfc1   = tf.matmul(do, tf.transpose(w4))
dflat  = tf.reshape(dfc1, [1, 4, 4, 256])

dpool3 = gen_nn_ops.avg_pool_grad(orig_input_shape=[1,8,8,256], grad=dflat, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
dact3  = dpool3 * tf.cast(tf.abs(conv3) > 0.0, dtype=tf.float32)
dconv3 = tf.nn.conv2d_backprop_input(input_sizes=[1,8,8,192], filter=tf.sign(tf.concat([w3[0], -1.0 * w3[1]], axis=3)), out_backprop=dact3, strides=[1,1,1,1], padding='SAME')

dpool2 = gen_nn_ops.avg_pool_grad(orig_input_shape=[1,16,16,192], grad=dconv3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
dact2  = dpool2 * tf.cast(tf.abs(conv2) > 0.0, dtype=tf.float32)
dconv2 = tf.nn.conv2d_backprop_input(input_sizes=[1,16,16,128], filter=tf.sign(tf.concat([w2[0], -1.0 * w2[1]], axis=3)), out_backprop=dact2, strides=[1,1,1,1], padding='SAME')

dpool1 = gen_nn_ops.avg_pool_grad(orig_input_shape=[1,32,32,128], grad=dconv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
dact1  = dpool1 * tf.cast(tf.abs(conv1) > 0.0, dtype=tf.float32)
dconv1 = tf.nn.conv2d_backprop_input(input_sizes=[1,32,32,6], filter=tf.sign(tf.concat([w1[0], -1.0 * w1[1]], axis=3)), out_backprop=dact1, strides=[1,1,1,1], padding='SAME')

####################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

xs = np.reshape(x_train[0], [1,32,32,6])
ys = np.reshape(y_train[0], [1,10])

[grad, dpool1] = sess.run([grad, dpool1], feed_dict={x: xs, y: ys})
print (np.sum(np.absolute(grad - dpool1)) / np.sum(np.absolute(grad)))
print (angle_between(np.reshape(grad, -1), np.reshape(dpool1, -1)) * (180.0 / 3.14))
























 
