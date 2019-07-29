

import numpy as np
import tensorflow as tf
import keras

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

f1 = 64
f2 = 96
f3 = 128

conv1_weights = tf.get_variable("conv1_weights", [3,3,3, f1], dtype=tf.float32)
conv2_weights = tf.get_variable("conv2_weights", [3,3,f1,f2], dtype=tf.float32)
conv3_weights = tf.get_variable("conv3_weights", [3,3,f2,f3], dtype=tf.float32)

pred_weights = tf.get_variable("pred_weights", [4*4*f3,10], dtype=tf.float32)
pred_bias = tf.get_variable("pred_bias", [10], dtype=tf.float32)

####################################

conv1      = tf.nn.conv2d(x, conv1_weights, [1,1,1,1], 'SAME')
relu1      = tf.nn.relu(conv1)
conv1_pool = tf.nn.avg_pool(relu1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

conv2      = tf.nn.conv2d(conv1_pool, conv2_weights, [1,1,1,1], 'SAME')
relu2      = tf.nn.relu(conv2)
conv2_pool = tf.nn.avg_pool(relu2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

conv3      = tf.nn.conv2d(conv2_pool, conv3_weights, [1,1,1,1], 'SAME')
relu3      = tf.nn.relu(conv3)
conv3_pool = tf.nn.avg_pool(relu3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

pred_view = tf.reshape(conv3_pool, [-1, 4*4*f3])
pred = tf.matmul(pred_view, pred_weights) + pred_bias

####################################

predict = tf.argmax(pred, axis=1)
correct = tf.equal(predict, tf.argmax(y, 1))
sum_correct = tf.reduce_sum(tf.cast(correct, tf.float32))

loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=pred)
params = tf.trainable_variables()
grads = tf.gradients(loss, params)
grads_and_vars = zip(grads, params)
train = tf.train.AdamOptimizer(learning_rate=1e-2, epsilon=1.).apply_gradients(grads_and_vars)

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
    

params = sess.run(params, feed_dict={})
for p in params:
    print (np.shape(p))
    
dic = {}
dic['conv1_weights'] = params[0]
dic['conv2_weights'] = params[1]
dic['conv3_weights'] = params[2]
dic['pred_weights'] = params[3]
dic['pred_bias'] = params[4]
np.save('cifar10_weights', dic)

    
        
        
