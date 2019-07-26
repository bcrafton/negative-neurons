
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

# op = 'conv'
op = 'local'

####################################

f1 = 64
f2 = 96
f3 = 128

####################################

pred_weights = tf.get_variable("pred_weights", [4*4*f3,10], dtype=tf.float32)
pred_bias = tf.get_variable("pred_bias", [10], dtype=tf.float32)

if op == 'conv':
    w1p = tf.get_variable("w1p", [3,3,3, f1], dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, 0, np.infty))
    w1n = tf.get_variable("w1n", [3,3,3, f1], dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, 0, np.infty))
    w2p = tf.get_variable("w2p", [3,3,f1,f2], dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, 0, np.infty))
    w2n = tf.get_variable("w2n", [3,3,f1,f2], dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, 0, np.infty))
    w3p = tf.get_variable("w3p", [3,3,f2,f3], dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, 0, np.infty))
    w3n = tf.get_variable("w3n", [3,3,f2,f3], dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, 0, np.infty))

    b1p = tf.Variable(tf.zeros(shape=f1))
    b1n = tf.Variable(tf.zeros(shape=f1))
    b2p = tf.Variable(tf.zeros(shape=f2))
    b2n = tf.Variable(tf.zeros(shape=f2))
    b3p = tf.Variable(tf.zeros(shape=f3))
    b3n = tf.Variable(tf.zeros(shape=f3))

if op == 'local':
    limit = np.sqrt(6. / (3.*3.*3. + 3.*3.*f1))
    w1p_init = np.random.uniform(low=-limit, high=limit, size=(32, 32, 3*3*3, f1))
    w1n_init = np.random.uniform(low=-limit, high=limit, size=(32, 32, 3*3*3, f1))
    w1p = tf.Variable(w1p_init, dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, 0, np.infty))
    w1n = tf.Variable(w1n_init, dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, 0, np.infty))

    limit = np.sqrt(6. / (3.*3.*f1 + 3.*3.*f2))
    w2p_init = np.random.uniform(low=-limit, high=limit, size=(16, 16, 3*3*f1, f2))
    w2n_init = np.random.uniform(low=-limit, high=limit, size=(16, 16, 3*3*f1, f2))
    w2p = tf.Variable(w2p_init, dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, 0, np.infty))
    w2n = tf.Variable(w2n_init, dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, 0, np.infty))

    limit = np.sqrt(6. / (3.*3.*f2 + 3.*3.*f3))
    w3p_init = np.random.uniform(low=-limit, high=limit, size=(8, 8, 3*3*f2, f3))
    w3n_init = np.random.uniform(low=-limit, high=limit, size=(8, 8, 3*3*f2, f3))
    w3p = tf.Variable(w3p_init, dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, 0, np.infty))
    w3n = tf.Variable(w3n_init, dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, 0, np.infty))

    b1p = tf.Variable(tf.zeros(shape=[32,32,f1]))
    b1n = tf.Variable(tf.zeros(shape=[32,32,f1]))
    b2p = tf.Variable(tf.zeros(shape=[16,16,f2]))
    b2n = tf.Variable(tf.zeros(shape=[16,16,f2]))
    b3p = tf.Variable(tf.zeros(shape=[ 8, 8,f3]))
    b3n = tf.Variable(tf.zeros(shape=[ 8, 8,f3]))

####################################

if op == 'conv':
    def conv_op(x, w):
        conv = tf.nn.conv2d(x, w, [1,1,1,1], 'SAME')
        return conv
        
if op == 'local':
    def conv_op(x, w):
        patches = tf.image.extract_image_patches(images=x, ksizes=[1,3,3,1], strides=[1,1,1,1], padding='SAME', rates=[1,1,1,1])
        patches = tf.transpose(patches, [1, 2, 0, 3])
        
        local   = tf.keras.backend.batch_dot(patches, w)
        local   = tf.transpose(local, [2, 0, 1, 3])
        
        return local

####################################

if f == 'relu(W_p @ x - W_n @ x)':
    def conv_block(x, wp, wn, bp, bn):
        convp = conv_op(x, wp) + bp
        convn = conv_op(x, wn) + bn
        conv  = tf.nn.relu(convp - convn)
        return conv

elif f == 'relu(W_p @ x) - relu(W_n @ x)':
    def conv_block(x, wp, wn, bp, bn):
        convp = conv_op(x, wp) + bp
        convn = conv_op(x, wn) + bn
        conv  = tf.nn.relu(convp) - tf.nn.relu(convn)
        return conv

elif f == 'relu(W_p @ x - relu(W_n @ x))':
    def conv_block(x, wp, wn, bp, bn):
        convp = conv_op(x, wp) + bp
        convn = conv_op(x, wn) + bn
        conv  = tf.nn.relu(convp - tf.nn.relu(convn))
        return conv

elif f == 'max(relu(W_p @ x), relu(W_n @ x))':
    def conv_block(x, wp, wn, bp, bn):
        convp = tf.nn.relu(conv_op(x, wp) + bp)
        convn = tf.nn.relu(conv_op(x, wn) + bn)
        
        compp = tf.cast(tf.greater(convp, convn), dtype=tf.float32) * convp
        compn = tf.cast(tf.greater(convn, convp), dtype=tf.float32) * convn
        
        conv  = compp - compn
        return conv

elif f == 'y = relu(W_p @ x); y - relu(W_n @ y)':
    def conv_block(x, wp, wn, bp, bn):
        convp = tf.nn.relu(conv_op(x, wp) + bp)
        convn = tf.nn.relu(conv_op(x, wn) + bn)
        conv  = convp - convn
        return conv

####################################

conv1 = conv_block(x=x, wp=w1p, wn=w1n, bp=b1p, bn=b1n)
pool1 = tf.nn.avg_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

conv2 = conv_block(x=pool1, wp=w2p, wn=w2n, bp=b2p, bn=b2n)
pool2 = tf.nn.avg_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

conv3 = conv_block(x=pool2, wp=w3p, wn=w3n, bp=b3p, bn=b3n)
pool3 = tf.nn.avg_pool(conv3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

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
        
        
