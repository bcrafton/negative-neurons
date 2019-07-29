
import numpy as np
import tensorflow as tf
# import keras

from util.conv import conv

#########################

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

assert(np.shape(x_train) == (50000, 32, 32, 3))
x_train = x_train - np.mean(x_train, axis=0, keepdims=True)
x_train = x_train / np.std(x_train, axis=0, keepdims=True)
# y_train = keras.utils.to_categorical(y_train, 10)

#########################

# def conv(x, filters, strides, padding):

x = np.reshape(x_train[0], [1, 32, 32, 3])
f = np.random.uniform(low=-1., high=1., size=(3,3,3,32))
s = [1, 1]
p = 'same'

y_ref = conv(x, f, s, p)

#########################

x = np.reshape(x_train[0], [1, 32, 32, 3])
fp = f * (f > 0.)
fn = np.absolute(f) * (f < 0.)
s = [1, 1]
p = 'same'

y_neg = conv(x, fp, s, p) - conv(x, fn, s, p)

#########################

print (np.all((y_ref - y_neg) < 1e-3))

#########################
