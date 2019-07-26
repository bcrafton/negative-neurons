


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

images = x
# images = tf.image.random_crop(images, [50, 24, 24, 3])
# images = tf.image.resize_images(images, [32, 32])

####################################

f1 = 64
f2 = 96
f3 = 128

limit = np.sqrt(6. / (3.*3.*3. + 3.*3.*f1))
l1w = np.random.uniform(low=-limit, high=limit, size=(32, 32, 3*3*3, f1))
local1_weights = tf.Variable(l1w, dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, 0, np.infty))

limit = np.sqrt(6. / (3.*3.*f1 + 3.*3.*f2))
l2w = np.random.uniform(low=-limit, high=limit, size=(16, 16, 3*3*(f1//2), f2))
local2_weights = tf.Variable(l2w, dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, 0, np.infty))

limit = np.sqrt(6. / (3.*3.*f2 + 3.*3.*f3))
l3w = np.random.uniform(low=-limit, high=limit, size=(8, 8, 3*3*(f2//2), f3))
local3_weights = tf.Variable(l3w, dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, 0, np.infty))

local1_bias = tf.constant(0.0)
local2_bias = tf.constant(0.0)
local3_bias = tf.constant(0.0)

'''
local1_bias = tf.Variable(np.zeros(shape=(32, 32, f1)), dtype=tf.float32)
local2_bias = tf.Variable(np.zeros(shape=(16, 16, f2)), dtype=tf.float32)
local3_bias = tf.Variable(np.zeros(shape=( 8,  8, f3)), dtype=tf.float32)
'''

'''
local1_mask = tf.constant(np.sign(np.random.uniform(low=-1, high=1, size=f1)), dtype=tf.float32)
local2_mask = tf.constant(np.sign(np.random.uniform(low=-1, high=1, size=f2)), dtype=tf.float32)
local3_mask = tf.constant(np.sign(np.random.uniform(low=-1, high=1, size=f3)), dtype=tf.float32)
'''

local1_mask = tf.constant(np.array([1] * (f1 // 2) + [-1] * (f1 // 2)), dtype=tf.float32)
local2_mask = tf.constant(np.array([1] * (f2 // 2) + [-1] * (f2 // 2)), dtype=tf.float32)
local3_mask = tf.constant(np.array([1] * (f3 // 2) + [-1] * (f3 // 2)), dtype=tf.float32)

pred_weights = tf.get_variable("pred_weights", [4*4*f3,10], dtype=tf.float32)
pred_bias = tf.get_variable("pred_bias", [10], dtype=tf.float32)

####################################

patches1 = tf.image.extract_image_patches(images=images, ksizes=[1,3,3,1], strides=[1,1,1,1], padding='SAME', rates=[1,1,1,1])
patches1 = tf.transpose(patches1, [1, 2, 0, 3])
local1   = tf.nn.relu(tf.keras.backend.batch_dot(patches1, local1_weights))
local1   = tf.transpose(local1, [2, 0, 1, 3])
local1   = (local1 + local1_bias) * local1_mask
local1   = local1[:, :, :, 0:f1//2] + local1[:, :, :, f1//2:f1]
local1_pool = tf.nn.avg_pool(local1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

patches2 = tf.image.extract_image_patches(images=local1_pool, ksizes=[1,3,3,1], strides=[1,1,1,1], padding='SAME', rates=[1,1,1,1])
patches2 = tf.transpose(patches2, [1, 2, 0, 3])
local2   = tf.nn.relu(tf.keras.backend.batch_dot(patches2, local2_weights))
local2   = tf.transpose(local2, [2, 0, 1, 3])
local2   = (local2 + local2_bias) * local2_mask
local2   = local2[:, :, :, 0:f2//2] + local2[:, :, :, f2//2:f2]
local2_pool = tf.nn.avg_pool(local2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

patches3 = tf.image.extract_image_patches(images=local2_pool, ksizes=[1,3,3,1], strides=[1,1,1,1], padding='SAME', rates=[1,1,1,1])
patches3 = tf.transpose(patches3, [1, 2, 0, 3])
local3   = tf.nn.relu(tf.keras.backend.batch_dot(patches3, local3_weights))
local3   = tf.transpose(local3, [2, 0, 1, 3])
local3   = (local3 + local3_bias) * local3_mask
local3   = local3[:, :, :, 0:f3//2] + local3[:, :, :, f3//2:f3]
local3_pool = tf.nn.avg_pool(local3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

pred_view = tf.reshape(local3_pool, [-1, 4*4*f3])
pred = tf.matmul(pred_view, pred_weights) + pred_bias

####################################

loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=pred)

predict = tf.argmax(pred, axis=1)
correct = tf.equal(predict, tf.argmax(y, 1))
sum_correct = tf.reduce_sum(tf.cast(correct, tf.float32))

params = tf.trainable_variables()
grads = tf.gradients(loss, params)
grads_and_vars = zip(grads, params)

train = tf.train.AdamOptimizer(learning_rate=1e-2, epsilon=1.).apply_gradients(grads_and_vars)

####################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for ii in range(25):
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
        
        
