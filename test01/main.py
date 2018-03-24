#!/usr/bin/env python3
#-*-coding:utf-8-*-
import tensorflow as tf
import numpy as np

def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

x_data = np.linspace(-1,1,300, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0 , x_data.shape).astype(np.float32)
y_data = 5*np.square(x_data) +x_data - 0.5 + noise

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
output_layer = add_layer(l1, 10, 1, activation_function=None)
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - output_layer),
                     reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
init = tf.global_variables_initializer()  # 替换成这样就好

saver = tf.train.Saver()

load = 0

with tf.Session() as sess:
    if load == 1:
        saver.restore(sess, "save/save.ckpt")
    else:
        sess.run(init)
        for i in range(10000):
            # training
            sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
            if i % 50 == 0:
                # to see the step improvement
                print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
        save_path = saver.save(sess, "save/save.ckpt")
        print("Save to path: ", save_path)

    asd = input()
    asd = float(asd)
    asd = np.asarray([[asd]])
    output_layer_value = sess.run(output_layer, feed_dict={xs: asd})
    print(output_layer_value)
