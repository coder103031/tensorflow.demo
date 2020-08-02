import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # 只显示 warning 和 Error

import tensorflow as tf
import numpy as np
#tf.where 条件语句真返回A，条件语句假返回B
a = tf.constant([1, 2, 3, 1, 1])
b = tf.constant([0, 1, 3, 4, 5])
c = tf.where(tf.greater(a, b), a, b)#greater比较ab熟大
print("c:", c)
#random.RandomState.rand() 返回一个【0,1）之间的随机数
rdm = np.random.RandomState(seed=1)  #seed=常数每次生成随机数相同
a = rdm.rand()
b = rdm.rand(2, 3)
print("a:", a)
print("b:", b)
#np.vstack() 将两个数组按垂直方向叠加
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = np.vstack((a, b))
print("c:\n", c)
#np.mgrid[] .ravel np.c_[]可以用来生成网格坐标点
x,y = np.mgrid[1:3:1, 2:4:0.5] #起始点：结束点：步长
grid = np.c_[x.ravel(), y.ravel()]  #ravel 平铺成一维  np.c_使返回的间隔数值点配对
print("x:", x)
print('y:', y)
print('grid:\n', grid)
#指数衰减学习率 = 初始学习率*学习率衰减率^(当前轮数/多少轮衰减一次）
w = tf.Variable(tf.constant(5, dtype=tf.float32))

epoch = 40      #当前轮数
LR_BASE = 0.2  	   #初始学习率
LR_DECAY = 0.99     #学习率衰减率
LR_STEP = 1		   #多少轮衰减一次

for epoch in range(epoch):  # for epoch 定义顶层循环，表示对数据集循环epoch次，此例数据集数据仅有1个w,初始化时候constant赋值为5。
    lr = LR_BASE * LR_DECAY ** (epoch / LR_STEP)
    with tf.GradientTape() as tape:   # with结构到grads框起了梯度的计算过程。
        loss = tf.square(w + 1)
    grads = tape.gradient(loss, w)  # .gradient函数告知谁对谁求导

    w.assign_sub(lr * grads)      # .assign_sub 对变量做自减 即：w -= lr*grads
    print("After %s epoch,w is %f,loss is %f,lr is %f" % (epoch, w.numpy(), loss, lr))
