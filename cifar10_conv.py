


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

f = 'relu(W_p @ x - W_n @ x)'
# f = 'relu(W_p @ x) - relu(W_n @ x)'
# f = 'relu(W_p @ x - relu(W_n @ x))'
# f = 'max(relu(W_p @ x), relu(W_n @ x))'

####################################

f1 = 96
f2 = 128
f3 = 256

conv1_weights_p = tf.get_variable("conv1_weights_p", [3,3,3, f1],   dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, 0, np.infty))
conv1_weights_n = tf.get_variable("conv1_weights_n", [3,3,3, f1],   dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, 0, np.infty))
conv2_weights_p = tf.get_variable("conv2_weights_p", [3,3,f1,f2],  dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, 0, np.infty))
conv2_weights_n = tf.get_variable("conv2_weights_n", [3,3,f1,f2],  dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, 0, np.infty))
conv3_weights_p = tf.get_variable("conv3_weights_p", [3,3,f2,f3], dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, 0, np.infty))
conv3_weights_n = tf.get_variable("conv3_weights_n", [3,3,f2,f3], dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, 0, np.infty))

pred_weights = tf.get_variable("pred_weights", [4*4*f3,10], dtype=tf.float32)
pred_bias = tf.get_variable("pred_bias", [10], dtype=tf.float32)

####################################

if f == 'relu(W_p @ x - W_n @ x)':
    conv1p     = tf.nn.conv2d(x, conv1_weights_p, [1,1,1,1], 'SAME')
    conv1n     = tf.nn.conv2d(x, conv1_weights_n, [1,1,1,1], 'SAME')
    conv1      = tf.nn.relu(conv1p - conv1n)
    conv1_pool = tf.nn.avg_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    conv2p     = tf.nn.conv2d(conv1_pool, conv2_weights_p, [1,1,1,1], 'SAME')
    conv2n     = tf.nn.conv2d(conv1_pool, conv2_weights_n, [1,1,1,1], 'SAME')
    conv2      = tf.nn.relu(conv2p - conv2n)
    conv2_pool = tf.nn.avg_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    conv3p     = tf.nn.conv2d(conv2_pool, conv3_weights_p, [1,1,1,1], 'SAME')
    conv3n     = tf.nn.conv2d(conv2_pool, conv3_weights_n, [1,1,1,1], 'SAME')
    conv3      = tf.nn.relu(conv3p - conv3n)
    conv3_pool = tf.nn.avg_pool(conv3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

elif f == 'relu(W_p @ x) - relu(W_n @ x)':
    conv1p     = tf.nn.conv2d(x, conv1_weights_p, [1,1,1,1], 'SAME')
    conv1n     = tf.nn.conv2d(x, conv1_weights_n, [1,1,1,1], 'SAME')
    conv1      = tf.nn.relu(conv1p) - tf.nn.relu(conv1n)
    conv1_pool = tf.nn.avg_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    conv2p     = tf.nn.conv2d(conv1_pool, conv2_weights_p, [1,1,1,1], 'SAME')
    conv2n     = tf.nn.conv2d(conv1_pool, conv2_weights_n, [1,1,1,1], 'SAME')
    conv2      = tf.nn.relu(conv2p) - tf.nn.relu(conv2n)
    conv2_pool = tf.nn.avg_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    conv3p     = tf.nn.conv2d(conv2_pool, conv3_weights_p, [1,1,1,1], 'SAME')
    conv3n     = tf.nn.conv2d(conv2_pool, conv3_weights_n, [1,1,1,1], 'SAME')
    conv3      = tf.nn.relu(conv3p) - tf.nn.relu(conv3n)
    conv3_pool = tf.nn.avg_pool(conv3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

elif f == 'relu(W_p @ x - relu(W_n @ x))':
    conv1p     = tf.nn.conv2d(x, conv1_weights_p, [1,1,1,1], 'SAME')
    conv1n     = tf.nn.conv2d(x, conv1_weights_n, [1,1,1,1], 'SAME')
    conv1      = tf.nn.relu(conv1p - tf.nn.relu(conv1n)) 
    conv1_pool = tf.nn.avg_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    conv2p     = tf.nn.conv2d(conv1_pool, conv2_weights_p, [1,1,1,1], 'SAME')
    conv2n     = tf.nn.conv2d(conv1_pool, conv2_weights_n, [1,1,1,1], 'SAME')
    conv2      = tf.nn.relu(conv2p - tf.nn.relu(conv2n))
    conv2_pool = tf.nn.avg_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    conv3p     = tf.nn.conv2d(conv2_pool, conv3_weights_p, [1,1,1,1], 'SAME')
    conv3n     = tf.nn.conv2d(conv2_pool, conv3_weights_n, [1,1,1,1], 'SAME')
    conv3      = tf.nn.relu(conv3p - tf.nn.relu(conv3n))
    conv3_pool = tf.nn.avg_pool(conv3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
  
elif f == 'max(relu(W_p @ x), relu(W_n @ x))':
    conv1p     = tf.nn.relu(tf.nn.conv2d(x, conv1_weights_p, [1,1,1,1], 'SAME'))
    conv1n     = tf.nn.relu(tf.nn.conv2d(x, conv1_weights_n, [1,1,1,1], 'SAME'))
    comp1p     = tf.cast(tf.greater(conv1p, conv1n), dtype=tf.float32) * conv1p
    comp1n     = tf.cast(tf.greater(conv1n, conv1p), dtype=tf.float32) * conv1n
    conv1      = comp1p - comp1n
    conv1_pool = tf.nn.avg_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    conv2p     = tf.nn.relu(tf.nn.conv2d(conv1_pool, conv2_weights_p, [1,1,1,1], 'SAME'))
    conv2n     = tf.nn.relu(tf.nn.conv2d(conv1_pool, conv2_weights_n, [1,1,1,1], 'SAME'))
    comp2p     = tf.cast(tf.greater(conv2p, conv2n), dtype=tf.float32) * conv2p
    comp2n     = tf.cast(tf.greater(conv2n, conv2p), dtype=tf.float32) * conv2n
    conv2      = comp2p - comp2n
    conv2_pool = tf.nn.avg_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    conv3p     = tf.nn.relu(tf.nn.conv2d(conv2_pool, conv3_weights_p, [1,1,1,1], 'SAME'))
    conv3n     = tf.nn.relu(tf.nn.conv2d(conv2_pool, conv3_weights_n, [1,1,1,1], 'SAME'))
    comp3p     = tf.cast(tf.greater(conv3p, conv3n), dtype=tf.float32) * conv3p 
    comp3n     = tf.cast(tf.greater(conv3n, conv3p), dtype=tf.float32) * conv3n
    conv3      = comp3p - comp3n
    conv3_pool = tf.nn.avg_pool(conv3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

####################################

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
        
        
