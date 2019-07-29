
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

w = np.load('cifar10_weights.npy').item()
tconv1_weights = w['conv1_weights']
tconv2_weights = w['conv2_weights']
tconv3_weights = w['conv3_weights']

tw1 = tf.Variable(tconv1_weights, dtype=tf.float32)
tw2 = tf.Variable(tconv2_weights, dtype=tf.float32)
tw3 = tf.Variable(tconv3_weights, dtype=tf.float32)

conv1_weights = np.random.normal(loc=np.average(tconv1_weights), scale=np.std(tconv1_weights), size=np.shape(tconv1_weights))
conv2_weights = np.random.normal(loc=np.average(tconv2_weights), scale=np.std(tconv2_weights), size=np.shape(tconv2_weights))
conv3_weights = np.random.normal(loc=np.average(tconv3_weights), scale=np.std(tconv3_weights), size=np.shape(tconv3_weights))

w1 = tf.Variable(conv1_weights, dtype=tf.float32)
w2 = tf.Variable(conv2_weights, dtype=tf.float32)
w3 = tf.Variable(conv3_weights, dtype=tf.float32)

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

conv3 = conv_op(pool2, w3)
pool3 = tf.nn.avg_pool(conv3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

####################################

tconv1 = conv_op(x, tw1)
tpool1 = tf.nn.avg_pool(tconv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

tconv2 = conv_op(tpool1, tw2)
tpool2 = tf.nn.avg_pool(tconv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

tconv3 = conv_op(tpool2, tw3)
tpool3 = tf.nn.avg_pool(tconv3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

####################################

loss = tf.losses.mean_squared_error(labels=tpool3, predictions=pool3)
params = [w1, w2, w3]
grads = tf.gradients(loss, params)
grads_and_vars = zip(grads, params)

train = tf.train.AdamOptimizer(learning_rate=10., epsilon=1.).apply_gradients(grads_and_vars)

####################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for ii in range(50):
    
    losses = []
    sign_matches = []
    
    for jj in range(0, 50000, 50):
        s = jj
        e = jj + 50
        xs = x_train[s:e]
        ys = y_train[s:e]
        
        [l, o, to, _] = sess.run([loss, pool3, tpool3, train], feed_dict={x: xs, y: ys})
        
        sign_match = np.sum(np.sign(o) == np.sign(to)) * 1.0 / np.prod(np.shape(o))
        sign_matches.append(sign_match)
        
        losses.append(l)
        
    print ('loss %f | match %f' % (np.average(losses), np.average(sign_matches)))
        
        
        
