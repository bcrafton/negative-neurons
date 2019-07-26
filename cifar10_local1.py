
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
# f = 'y = relu(W_p @ x); y - relu(W_n @ y)'

####################################

f1 = 64
f2 = 64
f3 = 64

local1_weights_p = tf.get_variable("local1_weights_p", [32,32,3*3*3, f1], dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, 0, np.infty))
local1_weights_n = tf.get_variable("local1_weights_n", [32,32,3*3*3, f1], dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, 0, np.infty))
local2_weights_p = tf.get_variable("local2_weights_p", [16,16,3*3*f1,f2], dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, 0, np.infty))
local2_weights_n = tf.get_variable("local2_weights_n", [16,16,3*3*f1,f2], dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, 0, np.infty))
local3_weights_p = tf.get_variable("local3_weights_p", [8, 8, 3*3*f2,f3], dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, 0, np.infty))
local3_weights_n = tf.get_variable("local3_weights_n", [8, 8, 3*3*f2,f3], dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, 0, np.infty))

conv1_bias_p = tf.Variable(tf.zeros(shape=[32,32,f1]))
conv1_bias_n = tf.Variable(tf.zeros(shape=[32,32,f1]))
conv2_bias_p = tf.Variable(tf.zeros(shape=[16,16,f2]))
conv2_bias_n = tf.Variable(tf.zeros(shape=[16,16,f2]))
conv3_bias_p = tf.Variable(tf.zeros(shape=[ 8, 8,f3]))
conv3_bias_n = tf.Variable(tf.zeros(shape=[ 8, 8,f3]))

pred_weights = tf.get_variable("pred_weights", [4*4*f3,10], dtype=tf.float32)
pred_bias = tf.get_variable("pred_bias", [10], dtype=tf.float32)

####################################

def local_op(x, w):
    patches = tf.image.extract_image_patches(images=x, ksizes=[1,3,3,1], strides=[1,1,1,1], padding='SAME', rates=[1,1,1,1])
    patches = tf.transpose(patches, [1, 2, 0, 3])
    
    local   = tf.keras.backend.batch_dot(patches, w)
    local   = tf.transpose(local, [2, 0, 1, 3])
    
    return local

####################################

if f == 'relu(W_p @ x - W_n @ x)':
    def local_block(x, wp, wn, bp, bn):
        localp = local_op(x, wp) + bp
        localn = local_op(x, wn) + bn
        local  = tf.nn.relu(localp - localn)
        return local

elif f == 'relu(W_p @ x) - relu(W_n @ x)':
    def local_block(x, wp, wn, bp, bn):
        localp = local_op(x, wp) + bp
        localn = local_op(x, wn) + bn
        local  = tf.nn.relu(localp) - tf.nn.relu(localn)
        return local

elif f == 'relu(W_p @ x - relu(W_n @ x))':
    def local_block(x, wp, wn, bp, bn):
        localp = local_op(x, wp) + bp
        localn = local_op(x, wn) + bn
        local  = tf.nn.relu(localp - tf.nn.relu(localn))
        return local

elif f == 'max(relu(W_p @ x), relu(W_n @ x))':
    def local_block(x, wp, wn, bp, bn):
        localp = tf.nn.relu(local_op(x, wp) + bp)
        localn = tf.nn.relu(local_op(x, wn) + bn)

        compp = tf.cast(tf.greater(localp, localn), dtype=tf.float32) * localp
        compn = tf.cast(tf.greater(localn, localp), dtype=tf.float32) * localn
        
        local  = compp - compn
        return local

elif f == 'y = relu(W_p @ x); y - relu(W_n @ y)':
    def local_block(x, wp, wn, bp, bn):
        localp = tf.nn.relu(local_op(x,      wp) + bp)
        localn = tf.nn.relu(local_op(localp, wn) + bn)
        local  = localp - localn
        return local
        
####################################

local1 = local_block(x=x, wp=local1_weights_p, wn=local1_weights_n, bp=conv1_bias_p, bn=conv1_bias_n)
pool1  = tf.nn.avg_pool(local1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

local2 = local_block(x=pool1, wp=local2_weights_p, wn=local2_weights_n, bp=conv2_bias_p, bn=conv2_bias_n)
pool2  = tf.nn.avg_pool(local2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

local3 = local_block(x=pool2, wp=local3_weights_p, wn=local3_weights_n, bp=conv3_bias_p, bn=conv3_bias_n)
pool3  = tf.nn.avg_pool(local3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

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
        
        
