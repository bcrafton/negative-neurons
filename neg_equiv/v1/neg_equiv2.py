
import numpy as np
import tensorflow as tf
# import keras

from util.conv import conv
from util.activation import relu

#########################

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

assert(np.shape(x_train) == (50000, 32, 32, 3))
x_train = x_train - np.mean(x_train, axis=0, keepdims=True)
x_train = x_train / np.std(x_train, axis=0, keepdims=True)
# y_train = keras.utils.to_categorical(y_train, 10)

#########################

x = np.reshape(x_train[0], [1, 32, 32, 3])
s = [1, 1]
p = 'same'

#########################

f = np.random.uniform(low=-1., high=1., size=(3,3,3,32))
y_ref = (conv(x, f, s, p))

#########################

xp = (x > 0.) * x
xn = (x < 0.) * np.absolute(x)

fp = (f > 0.) * f
fn = (f < 0.) * np.absolute(f)

y_neg = conv(xp, fp, s, p) - conv(xn, fp, s, p) - conv(xp, fn, s, p) + conv(xn, fn, s, p)

#########################

print (np.all((y_ref - y_neg) < 1e-3))
print (np.sum(np.sign(y_ref) == np.sign(y_neg)) * 1.0 / np.prod(np.shape(y_ref)))

#########################
