

import numpy as np
import tensorflow as tf
import keras

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

assert(np.shape(x_train) == (50000, 32, 32, 3))
# x_train = x_train - np.mean(x_train, axis=0, keepdims=True)
x_train = x_train / np.std(x_train, axis=0, keepdims=True)
x_train = np.concatenate((x_train, -1. * x_train), axis=3)
y_train = keras.utils.to_categorical(y_train, 10)

assert(np.shape(x_test) == (10000, 32, 32, 3))
# x_test = x_test - np.mean(x_test, axis=0, keepdims=True)
x_test = x_test / np.std(x_test, axis=0, keepdims=True)
x_test = np.concatenate((x_test, -1. * x_test), axis=3)
y_test = keras.utils.to_categorical(y_test, 10)

####################################

f0 = 6
f1 = 64
f2 = 96
f3 = 128

####################################

epochs = 50
batch_size = 50
x = tf.placeholder(tf.float32, [None, 32 , 32 , f0])
y = tf.placeholder(tf.float32, [None, 10])

####################################

w1 = tf.get_variable("w1", [3,3,f0,f1], dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, 0, np.infty))
w2 = tf.get_variable("w2", [3,3,f1,f2], dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, 0, np.infty))
w3 = tf.get_variable("w3", [3,3,f2,f3], dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, 0, np.infty))

w4 = tf.get_variable("fc1", [4*4*f3,10], dtype=tf.float32)

####################################

def conv_op(x, w):
    conv = tf.nn.conv2d(x, w, [1,1,1,1], 'SAME')
    relu = tf.nn.relu(conv)
    bn   = tf.layers.batch_normalization(relu)
    return bn

####################################

conv1 = conv_op(x, w1)
pool1 = tf.nn.avg_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

conv2 = conv_op(pool1, w2)
pool2 = tf.nn.avg_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

conv3 = conv_op(pool2, w3)
pool3 = tf.nn.avg_pool(conv3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

flat = tf.reshape(pool3, [batch_size, 4*4*f3])
fc1 = tf.matmul(flat, w4)

####################################

predict = tf.argmax(fc1, axis=1)
correct = tf.equal(predict, tf.argmax(y, 1))
sum_correct = tf.reduce_sum(tf.cast(correct, tf.float32))

loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=fc1)

[p1, p2, p3, p4, pg1, pb1, pg2, pb2, pg3, pb3] = tf.trainable_variables()

params1 = [w1, w4, pg1, pb1]
grads1 = tf.gradients(loss, params1)
grads_and_vars1 = zip(grads1, params1)
train1 = tf.train.AdamOptimizer(learning_rate=1e-3, epsilon=1.).apply_gradients(grads_and_vars1)

params2 = [w1, w2, w4, pg1, pb1, pg2, pb2]
grads2 = tf.gradients(loss, params2)
grads_and_vars2 = zip(grads2, params2)
train2 = tf.train.AdamOptimizer(learning_rate=1e-3, epsilon=1.).apply_gradients(grads_and_vars2)

params = tf.trainable_variables()
grads = tf.gradients(loss, params)
grads_and_vars = zip(grads, params)
train = tf.train.AdamOptimizer(learning_rate=1e-3, epsilon=1.).apply_gradients(grads_and_vars)

####################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

'''
[_params] = sess.run([params], feed_dict={})
for p in _params:
    print (np.shape(p))
'''

for ii in range(epochs):
    for jj in range(0, 50000, batch_size):
        s = jj
        e = jj + batch_size
        xs = x_train[s:e]
        ys = y_train[s:e]
        
        if ii < 10:
            sess.run([train1], feed_dict={x: xs, y: ys})
        elif ii < 20:
            sess.run([train2], feed_dict={x: xs, y: ys})
        else:
            sess.run([train], feed_dict={x: xs, y: ys})

    total_correct = 0
    for jj in range(0, 10000, batch_size):
        s = jj
        e = jj + batch_size
        xs = x_test[s:e]
        ys = y_test[s:e]
        _sum_correct = sess.run(sum_correct, feed_dict={x: xs, y: ys})
        total_correct += _sum_correct
            
    print ("acc: " + str(total_correct * 1.0 / 10000))
        
        
