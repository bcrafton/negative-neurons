
import argparse
import os
import sys

##############################################

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=250)
parser.add_argument('--batch_size', type=int, default=25)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--eps', type=float, default=1e-6)
args = parser.parse_args()

if args.gpu >= 0:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

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
lr = tf.placeholder(tf.float32, ())

####################################

w = np.load('cifar10_weights.npy', allow_pickle=True).item()
ref_w1_init = w['conv1_weights']
ref_w2_init = w['conv2_weights']
ref_w3_init = w['conv3_weights']

ref_w1 = tf.Variable(ref_w1_init, dtype=tf.float32)
ref_w2 = tf.Variable(ref_w2_init, dtype=tf.float32)
ref_w3 = tf.Variable(ref_w3_init, dtype=tf.float32)

####################################

ctrl_w1_init = np.random.normal(loc=np.average(ref_w1_init), scale=np.std(ref_w1_init), size=np.shape(ref_w1_init))
ctrl_w2_init = np.random.normal(loc=np.average(ref_w2_init), scale=np.std(ref_w2_init), size=np.shape(ref_w2_init))
ctrl_w3_init = np.random.normal(loc=np.average(ref_w3_init), scale=np.std(ref_w3_init), size=np.shape(ref_w3_init))

ctrl_w1 = tf.Variable(ctrl_w1_init, dtype=tf.float32)
ctrl_w2 = tf.Variable(ctrl_w2_init, dtype=tf.float32)
ctrl_w3 = tf.Variable(ctrl_w3_init, dtype=tf.float32)

####################################

w1p_init = np.random.normal(loc=np.average(ref_w1_init), scale=np.std(ref_w1_init), size=np.shape(ref_w1_init))
w1n_init = np.random.normal(loc=np.average(ref_w1_init), scale=np.std(ref_w1_init), size=np.shape(ref_w1_init))
w2p_init = np.random.normal(loc=np.average(ref_w2_init), scale=np.std(ref_w2_init), size=np.shape(ref_w2_init))
w2n_init = np.random.normal(loc=np.average(ref_w2_init), scale=np.std(ref_w2_init), size=np.shape(ref_w2_init))
w3_init  = np.random.normal(loc=np.average(ref_w3_init), scale=np.std(ref_w3_init), size=np.shape(ref_w3_init))

w1p = tf.Variable(w1p_init, dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, 0, np.infty))
w1n = tf.Variable(w1n_init, dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, 0, np.infty))
w2p = tf.Variable(w2p_init, dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, 0, np.infty))
w2n = tf.Variable(w2n_init, dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, 0, np.infty))
w3  = tf.Variable(w3_init,  dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, 0, np.infty))

####################################

def conv_op(x, w):
    conv = tf.nn.conv2d(x, w, [1,1,1,1], 'SAME')
    conv = tf.nn.relu(conv)
    return conv
    
def conv_op_np(x, wp, wn):
    convp = conv_op(x, wp)
    convn = conv_op(x, wn)
    conv  = tf.nn.relu(convp) - tf.nn.relu(convn)
    return conv

####################################

ref_conv1 = conv_op(x, ref_w1)
ref_pool1 = tf.nn.avg_pool(ref_conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

ref_conv2 = conv_op(ref_pool1, ref_w2)
ref_pool2 = tf.nn.avg_pool(ref_conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

ref_conv3 = conv_op(ref_pool2, ref_w3)
ref_pool3 = tf.nn.avg_pool(ref_conv3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

####################################

conv1 = conv_op_np(x, w1p, w1n)
pool1 = tf.nn.avg_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

conv2 = conv_op_np(pool1, w2p, w2n)
pool2 = tf.nn.avg_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

conv3 = tf.nn.relu(tf.nn.conv2d(pool2, w3, [1,1,1,1], 'SAME'))
pool3 = tf.nn.avg_pool(conv3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

loss = tf.losses.mean_squared_error(labels=ref_pool3, predictions=pool3)
params = [w1p, w1n, w2p, w2n, w3]
grads = tf.gradients(loss, params)
grads_and_vars = zip(grads, params)
train = tf.train.AdamOptimizer(learning_rate=lr, epsilon=args.eps).apply_gradients(grads_and_vars)

####################################

ctrl_conv1 = conv_op(x, ctrl_w1)
ctrl_pool1 = tf.nn.avg_pool(ctrl_conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

ctrl_conv2 = conv_op(ctrl_pool1, ctrl_w2)
ctrl_pool2 = tf.nn.avg_pool(ctrl_conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

ctrl_conv3 = conv_op(ctrl_pool2, ctrl_w3)
ctrl_pool3 = tf.nn.avg_pool(ctrl_conv3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

ctrl_loss = tf.losses.mean_squared_error(labels=ref_pool3, predictions=ctrl_pool3)
ctrl_params = [ctrl_w1, ctrl_w2, ctrl_w3]
ctrl_grads = tf.gradients(ctrl_loss, ctrl_params)
ctrl_grads_and_vars = zip(ctrl_grads, ctrl_params)
ctrl_train = tf.train.AdamOptimizer(learning_rate=lr, epsilon=args.eps).apply_gradients(ctrl_grads_and_vars)

####################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

####################################

random_losses = []
ctrl_random_losses = []

for jj in range(0, 50000, args.batch_size):
    s = jj
    e = jj + args.batch_size
    xs = x_train[s:e]
    ys = y_train[s:e]
    
    [l, cl] = sess.run([loss, ctrl_loss], feed_dict={x: xs, y: ys, lr: 0.0})
    
    random_losses.append(l)
    ctrl_random_losses.append(cl)

####################################

for ii in range(args.epochs):
    
    losses = []
    ctrl_losses = []
    
    for jj in range(0, 50000, args.batch_size):
        s = jj
        e = jj + args.batch_size
        xs = x_train[s:e]
        ys = y_train[s:e]
        
        [l, cl, _, _] = sess.run([loss, ctrl_loss, train, ctrl_train], feed_dict={x: xs, y: ys, lr: args.lr})
        
        losses.append(l)
        ctrl_losses.append(cl)
        
    print ('loss %f/%f | ctrl loss %f/%f' % (np.average(losses), np.average(random_losses), np.average(ctrl_losses), np.average(ctrl_random_losses)))
        
####################################
        
