
import numpy as np
import tensorflow as tf
import keras

####################################

x      = tf.placeholder(tf.float32, [1,32,32,1])
w      = tf.placeholder(tf.float32, [3,3,1,1])

conv1 = tf.nn.conv2d(x,     w, [1,1,1,1], 'SAME')
pool1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

conv2 = tf.nn.conv2d(pool1, w, [1,1,1,1], 'SAME')
pool2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

conv3 = tf.nn.conv2d(pool2, w, [1,1,1,1], 'SAME')
pool3 = tf.nn.max_pool(conv3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

output1 = tf.reshape(pool1, [16, 16])
output2 = tf.reshape(pool2, [8, 8])
output3 = tf.reshape(pool3, [4, 4])

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

####################################

xin = np.zeros(shape=(1,32,32,1))
xin[0][4][4][0] = 1

win = np.ones(shape=(3,3,1,1))

[output1, output2, output3] = sess.run([output1, output2, output3], feed_dict={x: xin, w: win})

print (output1)
print (output2)
print (output3)

####################################







 
