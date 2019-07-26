
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

# f = 'relu(W_p @ x - W_n @ x)'
# f = 'relu(W_p @ x) - relu(W_n @ x)'
# f = 'relu(W_p @ x - relu(W_n @ x))'
# f = 'max(relu(W_p @ x), relu(W_n @ x))'
f = 'y = relu(W_p @ x); y - relu(W_n @ y)'

####################################

f1 = 96
f2 = 128
f3 = 256

conv1_weights_p = tf.get_variable("conv1_weights_p", [3,3,3, f1], dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, 0, np.infty))
conv1_weights_n = tf.get_variable("conv1_weights_n", [3,3,3, f1], dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, 0, np.infty))
conv2_weights_p = tf.get_variable("conv2_weights_p", [3,3,f1,f2], dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, 0, np.infty))
conv2_weights_n = tf.get_variable("conv2_weights_n", [3,3,f1,f2], dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, 0, np.infty))
conv3_weights_p = tf.get_variable("conv3_weights_p", [3,3,f2,f3], dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, 0, np.infty))
conv3_weights_n = tf.get_variable("conv3_weights_n", [3,3,f2,f3], dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, 0, np.infty))

conv1_bias_p = tf.Variable(tf.zeros(shape=f1))
conv1_bias_n = tf.Variable(tf.zeros(shape=f1))
conv2_bias_p = tf.Variable(tf.zeros(shape=f2))
conv2_bias_n = tf.Variable(tf.zeros(shape=f2))
conv3_bias_p = tf.Variable(tf.zeros(shape=f3))
conv3_bias_n = tf.Variable(tf.zeros(shape=f3))

pred_weights = tf.get_variable("pred_weights", [4*4*f3,10], dtype=tf.float32)
pred_bias = tf.get_variable("pred_bias", [10], dtype=tf.float32)

####################################

def conv_block(x, wp, wn, bp, bn):
    convp = tf.nn.conv2d(x, wp, [1,1,1,1], 'SAME') + bp
    convn = tf.nn.conv2d(x, wn, [1,1,1,1], 'SAME') + bn
    conv  = tf.nn.relu(convp - convn)
    return conv

####################################

conv1 = conv_block(x=x, wp=conv1_weights_p, wn=conv1_weights_n, bp=conv1_bias_p, bn=conv1_bias_n)
pool1 = tf.nn.avg_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

conv2 = conv_block(x=pool1, wp=conv2_weights_p, wn=conv2_weights_n, bp=conv2_bias_p, bn=conv2_bias_n)
pool2 = tf.nn.avg_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

conv3 = conv_block(x=pool2, wp=conv3_weights_p, wn=conv3_weights_n, bp=conv3_bias_p, bn=conv3_bias_n)
pool3 = tf.nn.avg_pool(conv3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

####################################

pred_view = tf.reshape(pool3, [-1, 4*4*f3])
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
        
        
