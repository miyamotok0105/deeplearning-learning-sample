# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

'''
モデル設定
'''
tf.set_random_seed(0)  # 乱数シード

#Variableはニューラルネット自体のパラメータなど保持
w = tf.Variable(tf.zeros([2, 1]))
b = tf.Variable(tf.zeros([1]))

#placeholderメモリ確保。
#placeholderはニューラルネットに入れるデータのデータ構造
x = tf.placeholder(tf.float32, shape=[None, 2])
t = tf.placeholder(tf.float32, shape=[None, 1])

print("x ", x)
print("t ", t)
#matmul掛け算
y = tf.nn.sigmoid(tf.matmul(x, w) + b)

print("y ", y)

#reduce_sum足し算
cross_entropy = - tf.reduce_sum(t * tf.log(y) + (1 - t) * tf.log(1 - y))
#GradientDescentOptimizer 勾配降下法
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
#equalイコール、greater大きい法？
correct_prediction = tf.equal(tf.to_float(tf.greater(y, 0.5)), t)
print("correct_prediction ", correct_prediction)

'''
モデル学習
'''
# ORゲート
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [1]])

# 初期化
sess = tf.Session()
sess.run(tf.initialize_all_variables())

# init = tf.global_variables_initializer()
# sess = tf.Session()
# sess.run(init)

# 学習
for epoch in range(200):
    print("epoch ", epoch)
    sess.run(train_step, feed_dict={x: X, t: Y})

'''
学習結果の確認
'''
classified = correct_prediction.eval(session=sess, feed_dict={x: X, t: Y})
prob = y.eval(session=sess, feed_dict={x: X})

print('classified:')
print(classified)
print()
print('output probability:')
print(prob)
