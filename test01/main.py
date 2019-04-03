#!/usr/bin/env python3
# -*-coding:utf-8-*-
import tensorflow as tf
import numpy as np


def add_layer(inputs,in_size,out_size,activation_function=None):
    w = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.random_normal([1, out_size]))
    z = tf.matmul(inputs, w) + biases
    if activation_function is None:
        outputs = z
    else:
        outputs = activation_function(z)
    return outputs


x_data = np.linspace(-10, 10, 100, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0, x_data.shape).astype(np.float32)

y_data = np.divide(np.abs(x_data), x_data)
print(y_data)
#y_data = 5*np.square(x_data) + x_data - 0.5  # + noise

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

l1 = add_layer(xs, 1, 5)
output_layer = add_layer(l1, 5, 1, activation_function=tf.nn.tanh)

loss = tf.reduce_sum(tf.square(ys - output_layer))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
init = tf.global_variables_initializer()

saver = tf.train.Saver()

load = 0
train = 1
save = 1

last_loss = 1000000

with tf.Session() as sess:
    if load == 1:
        saver.restore(sess, "save/save.ckpt")

    if train == 1:
        if load != 1:
            sess.run(init)
        for i in range(100):
            # training
            sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
            if i % 50 == 0:
                # to see the step improvement
                l = sess.run(loss, feed_dict={xs: x_data, ys: y_data})
                print("loss:{}".format(l))
                if l < last_loss and save == 1 and i % 200 == 0:
                    last_loss= l
                    save_path = saver.save(sess, "save/save.ckpt")
                    print("Save to path: ", save_path)
        #save_path = saver.save(sess, "save/save.ckpt")
        #print("Save to path: ", save_path)
    while True:
        asd = np.asarray([[float(input())], [float(input())]])
        output_layer_value = sess.run(output_layer, feed_dict={xs: asd})
        print(output_layer_value)
