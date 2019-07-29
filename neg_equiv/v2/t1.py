
import numpy as np
import tensorflow as tf
import keras
from collections import deque

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

assert(np.shape(x_train) == (50000, 32, 32, 3))
x_train = x_train - np.mean(x_train, axis=0, keepdims=True)
x_train = x_train / np.std(x_train, axis=0, keepdims=True)
y_train = keras.utils.to_categorical(y_train, 10)

assert(np.shape(x_test) == (10000, 32, 32, 3))
x_test = x_test - np.mean(x_test, axis=0, keepdims=True)
x_test = x_test / np.std(x_test, axis=0, keepdims=True)
y_test = keras.utils.to_categorical(y_test, 10)

####################################

x = tf.placeholder(tf.float32, [None, 32 , 32 , 3])
y = tf.placeholder(tf.float32, [None, 10])

####################################

f1 = 16
f2 = 32

####################################

w1 = tf.get_variable("w1", [3,3,3, f1], dtype=tf.float32)
w2 = tf.get_variable("w2", [3,3,f1,f2], dtype=tf.float32)

tw1 = tf.get_variable("tw1", [3,3,3, f1], dtype=tf.float32)
tw2 = tf.get_variable("tw2", [3,3,f1,f2], dtype=tf.float32)

####################################

def conv_op(x, w):
    conv = tf.nn.conv2d(x, w, [1,1,1,1], 'SAME')
    conv = tf.nn.relu(conv)
    return conv

####################################

conv1 = conv_op(x, w1)
pool1 = tf.nn.avg_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

conv2 = conv_op(pool1, w2)
pool2 = tf.nn.avg_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

####################################

tconv1 = conv_op(x, tw1)
tpool1 = tf.nn.avg_pool(tconv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

tconv2 = conv_op(tpool1, tw2)
tpool2 = tf.nn.avg_pool(tconv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

####################################

loss = tf.losses.mean_squared_error(labels=tpool2, predictions=pool2)
params = [w1, w2]
grads = tf.gradients(loss, params)
grads_and_vars = zip(grads, params)

train = tf.train.AdamOptimizer(learning_rate=10., epsilon=1.).apply_gradients(grads_and_vars)
# train = tf.train.GradientDescentOptimizer(learning_rate=1.).apply_gradients(grads_and_vars)

####################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for ii in range(50):
    
    losses = []
    sign_matches = []
    nonzeros_target = []
    nonzeros = []
    
    for jj in range(0, 50000, 50):
        s = jj
        e = jj + 50
        xs = x_train[s:e]
        ys = y_train[s:e]
        [l, o, to, _] = sess.run([loss, pool2, tpool2, train], feed_dict={x: xs, y: ys})
        
        sign_match = np.sum(np.sign(o) == np.sign(to)) * 1.0 / np.prod(np.shape(o))
        sign_matches.append(sign_match)
        
        nonzero = np.count_nonzero(o) * 1.0 / np.prod(np.shape(o))
        nonzeros.append(nonzero)
        
        nonzero_target = np.count_nonzero(to) * 1.0 / np.prod(np.shape(to))
        nonzeros_target.append(nonzero_target)
        
        losses.append(l)
        
    [_w1, _w2, _tw1, _tw2] = sess.run([w1, w2, tw1, tw2], feed_dict={})
    sign_match_w1 = np.sum(np.sign(_w1) == np.sign(_tw1)) * 1.0 / np.prod(np.shape(_w1))
    sign_match_w2 = np.sum(np.sign(_w2) == np.sign(_tw2)) * 1.0 / np.prod(np.shape(_w2))
    
    print ('loss %f | match %f | nonzero %f %f | weight match %f %f' % (np.average(losses), np.average(sign_matches), np.average(nonzeros), np.average(nonzeros_target), sign_match_w1, sign_match_w2))
        
        
        
