import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # 只显示 warning 和 Error

import tensorflow as tf
import numpy as np

SEED = 23455
COST = 99
PROFIT = 1
rdm = np.random.RandomState(seed= SEED)   #生成0到1之间的随机数
x = rdm.rand(32, 2)
y_ = [[x1 + x2 + (rdm.rand() / 10.0 - 0.05)] for (x1, x2) in x] #生成噪声数据
x = tf.cast(x, dtype = tf.float32)

w1 = tf.Variable(tf.random.normal([2, 1], stddev=1, seed=1))

epoch = 15000
lr = 0.002

for epoch in range(epoch):
    with tf.GradientTape() as tape:
        y = tf.matmul(x, w1)
        loss_mse = tf.reduce_sum(tf.where(tf.greater(y, y_),(y - y_)* COST, (y_ - y) * PROFIT))

    grads = tape.gradient(loss_mse, w1)
    w1.assign_sub(lr * grads)

    if epoch % 500 ==0:
        print("After %d traing steps,w1 is"%(epoch))
        print(w1.numpy(),'\n')
print("Final w1 is:", w1.numpy())