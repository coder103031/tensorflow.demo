import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # 只显示 warning 和 Error

#张量（tensor）：多维数组（列表） 阶：张量的维数    几阶几个[
#维数    阶           名字                  例子
#0-D     0          标量 scalar          s= 1 2 3
#1-D     1          向量 vector          v=[1,2,3]
#2-D     2          矩阵 matrix          m=[[1,2,3],[4,5,6],[7,8,9]]
#n-D     n          张量tensor           t=[[[  n个[

import tensorflow as tf
import numpy as np
#如何创建一个张量 可用constant
a = tf.constant([1, 5],dtype=tf.int64)
print(a)
print(a.dtype)
print(a.shape)
#将numpy的数据转换为Tensor数据类型
a = np.arange(0, 5)
b = tf.convert_to_tensor(a, dtype = tf.int64)
print(a)
print(b)
#创建值全为0、1、9的张量
a = tf.zeros([2, 3])
b = tf.ones(4)
c = tf.fill([2, 2], 9)
print(a)
print(b)
print(c)
#生成正态分布的随机数，默认均值为0，标准差为1
d = tf.random.normal([2, 2], mean=0.5, stddev=1)
print(d)
#生成均匀分布的随机数（维度，最小值，最大值）
f = tf.random.uniform([2, 2], minval=0.5, maxval=1)
print(f)
#强制tensor转换数据类型，计算张量维度上元素的最大值和最小值
x1 = tf.constant([1., 2., 3.], dtype=tf.float64)
print(x1)
x2 = tf.cast(x1, tf.int32)
print(x2)
print(tf.reduce_min(x2),tf.reduce_max(x2))
#tensorflow中的数学运算
#对应元素的四则运算：tf.add, tf.subtract, tf.multiply, tf.divide
#平方、次方与开方：tf.square, tf.pow, tf.sqrt
#矩阵乘：tf.matmul

#输入特征和标签配对函数 tf.data.Dataset.from_tensor_slices
features = tf.constant([12, 23, 10, 17])
labels = tf.constant([0, 1, 1, 0])
dataset = tf.data.Dataset.from_tensor_slices((features, labels))
print(dataset)
for element in dataset:
    print(element)