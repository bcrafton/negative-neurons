
import numpy as np
import tensorflow as tf
# import keras

from util.conv import conv
from util.activation import relu

#########################

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

assert(np.shape(x_train) == (50000, 32, 32, 3))
# x_train = x_train - np.mean(x_train, axis=0, keepdims=True)
x_train = x_train / np.std(x_train, axis=0, keepdims=True)
# y_train = keras.utils.to_categorical(y_train, 10)

#########################

x = np.reshape(x_train[0], [1, 32, 32, 3])
s = [1, 1]
p = 'same'

#########################

f1 = np.random.uniform(low=-1., high=1., size=(3,3,3, 32))
f2 = np.random.uniform(low=-1., high=1., size=(3,3,32,48))

y_ref1 = relu(conv(x,      f1, s, p))
y_ref2 = conv(y_ref1, f2, s, p)

#########################

fp1 = (f1 > 0.) * f1
fn1 = (f1 < 0.) * np.absolute(f1)

fp2 = (f2 > 0.) * f2
fn2 = (f2 < 0.) * np.absolute(f2)

y_neg1 = relu(conv(x, fp1, s, p))      - relu(conv(x, fn1, s, p))
y_neg2 = relu(conv(y_neg1, fp2, s, p)) - relu(conv(y_neg1, fn2, s, p))

#########################

print (np.all((y_ref2 - y_neg2) < 1e-3))
print (np.sum(np.sign(y_ref2) == np.sign(y_neg2)) * 1.0 / np.prod(np.shape(y_ref2)))

#########################
