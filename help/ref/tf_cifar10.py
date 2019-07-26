

import numpy as np
import tensorflow as tf
import keras

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

assert(np.shape(x_train) == (50000, 32, 32, 3))
# x_train = x_train - np.mean(x_train, axis=0, keepdims=True)
x_train = x_train / np.std(x_train, axis=0, keepdims=True)
y_train = keras.utils.to_categorical(y_train, 10)

assert(np.shape(x_test) == (10000, 32, 32, 3))
# x_test = x_test - np.mean(x_test, axis=0, keepdims=True)
x_test = x_test / np.std(x_test, axis=0, keepdims=True)
y_test = keras.utils.to_categorical(y_test, 10)

####################################

x = tf.placeholder(tf.float32, [None, 32 , 32 , 3])
y = tf.placeholder(tf.float32, [None, 10])

####################################

local1_weights = tf.get_variable("local1_weights", [32,32,3*3*3, 32], dtype=tf.float32)
local2_weights = tf.get_variable("local2_weights", [16,16,3*3*32,32], dtype=tf.float32)
local3_weights = tf.get_variable("local3_weights", [8, 8, 3*3*32,32], dtype=tf.float32)

pred_weights = tf.get_variable("pred_weights", [4*4*32,10], dtype=tf.float32)
pred_bias = tf.get_variable("pred_bias", [10], dtype=tf.float32)

####################################

patches1 = tf.image.extract_image_patches(images=x, ksizes=[1,3,3,1], strides=[1,1,1,1], padding='SAME', rates=[1,1,1,1])
patches1 = tf.transpose(patches1, [1, 2, 0, 3])
local1   = tf.nn.relu(tf.keras.backend.batch_dot(patches1, local1_weights))
local1   = tf.transpose(local1, [2, 0, 1, 3])
local1_pool = tf.nn.avg_pool(local1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

patches2 = tf.image.extract_image_patches(images=local1_pool, ksizes=[1,3,3,1], strides=[1,1,1,1], padding='SAME', rates=[1,1,1,1])
patches2 = tf.transpose(patches2, [1, 2, 0, 3])
local2   = tf.nn.relu(tf.keras.backend.batch_dot(patches2, local2_weights))
local2   = tf.transpose(local2, [2, 0, 1, 3])
local2_pool = tf.nn.avg_pool(local2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

patches3 = tf.image.extract_image_patches(images=local2_pool, ksizes=[1,3,3,1], strides=[1,1,1,1], padding='SAME', rates=[1,1,1,1])
patches3 = tf.transpose(patches3, [1, 2, 0, 3])
local3   = tf.nn.relu(tf.keras.backend.batch_dot(patches3, local3_weights))
local3   = tf.transpose(local3, [2, 0, 1, 3])
local3_pool = tf.nn.avg_pool(local3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

pred_view = tf.reshape(local3_pool, [-1, 4*4*32])
pred = tf.matmul(pred_view, pred_weights) + pred_bias

####################################

loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=pred)

predict = tf.argmax(pred, axis=1)
correct = tf.equal(predict, tf.argmax(y, 1))
sum_correct = tf.reduce_sum(tf.cast(correct, tf.float32))

params = tf.trainable_variables()
grads = tf.gradients(loss, params)
grads_and_vars = zip(grads, params)

train = tf.train.AdamOptimizer(learning_rate=1e-2, epsilon=1e-1).apply_gradients(grads_and_vars)

####################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for ii in range(10):
    for jj in range(0, 50000, 50):
        s = jj
        e = jj + 50
        xs = x_train[s:e]
        ys = y_train[s:e]
        sess.run([train], feed_dict={x: xs, y: ys})
        
    total_correct = 0

    for jj in range(0, 10000, 50):
        s = jj
        e = jj + 50
        xs = x_test[s:e]
        ys = y_test[s:e]
        _sum_correct = sess.run(sum_correct, feed_dict={x: xs, y: ys})
        total_correct += _sum_correct
            
    print ("acc: " + str(total_correct * 1.0 / 10000))
        
        
