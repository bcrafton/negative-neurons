
import argparse
import os
import sys

##############################################

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--eps', type=float, default=1e-6)
args = parser.parse_args()

if args.gpu >= 0:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

import numpy as np
import tensorflow as tf
import keras
from collections import deque

(x_train, y_train), (_, _) = tf.keras.datasets.cifar10.load_data()

assert(np.shape(x_train) == (50000, 32, 32, 3))
x_train = x_train - np.mean(x_train, axis=0, keepdims=True)
x_train = x_train / np.std(x_train, axis=0, keepdims=True)
y_train = keras.utils.to_categorical(y_train, 10)

####################################

x = tf.placeholder(tf.float32, [None, 32 , 32 , 3])
y = tf.placeholder(tf.float32, [None, 10])
lr = tf.placeholder(tf.float32, ())

####################################

w = np.load('cifar10_weights.npy', allow_pickle=True).item()
ref_w1_init = w['conv1_weights'][:, :, :,   0:8]
ref_w2_init = w['conv2_weights'][:, :, 0:8, 0:16]

ref_w1 = tf.Variable(ref_w1_init, dtype=tf.float32)
ref_w2 = tf.Variable(ref_w2_init, dtype=tf.float32)

####################################

ctrl_w1_init = np.random.normal(loc=np.average(ref_w1_init), scale=np.std(ref_w1_init), size=np.shape(ref_w1_init))
ctrl_w2_init = np.random.normal(loc=np.average(ref_w2_init), scale=np.std(ref_w2_init), size=np.shape(ref_w2_init))

ctrl_w1 = tf.Variable(ctrl_w1_init, dtype=tf.float32)
ctrl_w2 = tf.Variable(ctrl_w2_init, dtype=tf.float32)

ctrl_b1 = tf.Variable(np.zeros(shape=np.shape(ref_w1_init)), dtype=tf.float32)
ctrl_b2 = tf.Variable(np.zeros(shape=np.shape(ref_w2_init)), dtype=tf.float32)

z1  = tf.constant(np.zeros(shape=np.shape(ref_w1_init)), dtype=tf.float32)
z2  = tf.constant(np.zeros(shape=np.shape(ref_w2_init)), dtype=tf.float32)

####################################

w1p_init = np.absolute(np.random.normal(loc=np.average(ref_w1_init), scale=np.std(ref_w1_init), size=[3, 3, 3, 8]))
w1n_init = np.absolute(np.random.normal(loc=np.average(ref_w1_init), scale=np.std(ref_w1_init), size=[3, 3, 3, 8]))
w2p_init = np.absolute(np.random.normal(loc=np.average(ref_w2_init), scale=np.std(ref_w2_init), size=[3, 3, 8, 16]))
w2n_init = np.absolute(np.random.normal(loc=np.average(ref_w2_init), scale=np.std(ref_w2_init), size=[3, 3, 8, 16]))

w1p = tf.Variable(w1p_init, dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, 0, np.infty))
w1n = tf.Variable(w1n_init, dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, 0, np.infty))
w2p = tf.Variable(w2p_init, dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, 0, np.infty))
w2n = tf.Variable(w2n_init, dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, 0, np.infty))

####################################

'''
def conv_op(x, f, b, szx, szf):
    
    num_batch   = szx[0]
    num_patch   = szx[1] * szx[2]
    h           = szx[1]
    w           = szx[2]
    
    dim_filter  = szf[0] * szf[1] * szf[2]
    num_filter  = szf[3]
    kh          = szf[0]
    kw          = szf[1]
    
    patches     = tf.image.extract_image_patches(images=x, ksizes=[1, kh, kw, 1], strides=[1,1,1,1], padding='SAME', rates=[1,1,1,1]) # [50, 32, 32, 27]
    patches     = tf.reshape(patches, (num_batch * num_patch, dim_filter))                                                            # [50*32*32, 27]
    patches     = tf.
    patches     = tf.cast(patches > b, dtype=tf.float32)
    
    f           = tf.reshape(f, [dim_filter, num_filter])                                                                             # [3, 3, 3, 32] -> [27, 32]
    
    conv        = tf.matmul(patches, f)                                                                                               # [50*32*32, 27] @ [27, 32] -> [50*32*32, 27]
    conv        = tf.reshape(conv, [num_batch, h, w, num_filter])
    
    return conv
'''

'''
def conv_op(x, w, b, szx, sxf):
    # x = tf.Print(x, [tf.shape(w)[0] * tf.shape(w)[1] * tf.shape(w)[2]], message='', summarize=1000)
    conv = tf.nn.conv2d(x, w, [1,1,1,1], 'SAME')
    conv = tf.nn.relu(conv)
    return conv
'''

# '''
def conv_op(x, f, b, szx, szf):
    
    num_batch   = szx[0]
    num_patch   = szx[1] * szx[2]
    h           = szx[1]
    w           = szx[2]
    
    dim_filter  = szf[0] * szf[1] * szf[2]
    num_filter  = szf[3]
    kh          = szf[0]
    kw          = szf[1]
    
    patches     = tf.image.extract_image_patches(images=x, ksizes=[1, kh, kw, 1], strides=[1,1,1,1], padding='SAME', rates=[1,1,1,1]) # [50, 32, 32, 27]
    patches     = tf.reshape(patches, (num_batch * num_patch, dim_filter))                                                            # [50*32*32, 27]
    
    f           = tf.reshape(f, [dim_filter, num_filter])                                                                             # [3, 3, 3, 32] -> [27, 32]
    
    conv        = tf.matmul(patches, f)                                                                                               # [50*32*32, 27] @ [27, 32] -> [50*32*32, 27]
    conv        = tf.reshape(conv, [num_batch, h, w, num_filter])
    
    return conv
# '''

def conv_op_np(x, wp, wn):
    convp = tf.nn.conv2d(x, wp, [1,1,1,1], 'SAME')
    convn = tf.nn.conv2d(x, wn, [1,1,1,1], 'SAME')
    
    # this is useless ... we need negative outputs.
    # conv  = tf.nn.relu(convp - tf.nn.relu(convn))
    
    # and this is retarded, we would have another set of weights and activation ... would not just subtract these
    # conv  = tf.nn.relu(convp) - tf.nn.relu(convn)
    
    conv = convp
    
    return conv

####################################

ref_conv1 = conv_op(x, ref_w1, z1, [50,32,32,3], [3,3,3,8])
ref_pool1 = tf.nn.avg_pool(ref_conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

ref_conv2 = conv_op(ref_pool1, z2, ref_w2, [50,16,16,8], [3,3,8,16])
ref_pool2 = tf.nn.avg_pool(ref_conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

####################################

conv1 = conv_op_np(x, w1p, w1n)
pool1 = tf.nn.avg_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

conv2 = conv_op_np(pool1, w2p, w2n)
pool2 = tf.nn.avg_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

loss = tf.losses.mean_squared_error(labels=ref_pool2, predictions=pool2)
params = [w1p, w1n, w2p, w2n]
grads = tf.gradients(loss, params)
grads_and_vars = zip(grads, params)
train = tf.train.AdamOptimizer(learning_rate=lr, epsilon=args.eps).apply_gradients(grads_and_vars)

####################################

ctrl_conv1 = conv_op(x, ctrl_w1, ctrl_b1, [50,32,32,3], [3,3,3,8])
ctrl_pool1 = tf.nn.avg_pool(ctrl_conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

ctrl_conv2 = conv_op(ctrl_pool1, ctrl_w2, ctrl_b2, [50,16,16,8], [3,3,8,16])
ctrl_pool2 = tf.nn.avg_pool(ctrl_conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

ctrl_loss = tf.losses.mean_squared_error(labels=ref_pool2, predictions=ctrl_pool2)
ctrl_params = [ctrl_w1, ctrl_w2]
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
        
