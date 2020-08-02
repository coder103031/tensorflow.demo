import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # 只显示 warning 和 Error

import tensorflow as tf
import numpy as np

#损失函数求值 使用with结构记录计算过程
with tf.GradientTape() as tape:
    x = tf.Variable(tf.constant(3.0))
    y = tf.pow(x, 2)
grad = tape.gradient(y, x) #对x求导
print(grad)
#enumerate枚举出所有元素并配上序号
seq = ['one', 'two', 'three']
for i,element in enumerate(seq):
    print(i, element)
#tf.one_hot（待转换数据，depth=几分类）独热编码
classes = 3
labels = tf.constant([1, 2, 0])  #输入的元素最小值为0，最大值为2
output = tf.one_hot(labels,depth = classes)
print(output)
#softmax=e^xi/∑e^xi,输出符合概率分布
y = tf.constant([1.01, 2.01, -0.66])
y_pro = tf.nn.softmax(y)
print("After softmax,y_pro is:",y_pro)
#参数自更新assign_sub 调用前先用tf.Variable定义变量w为可训练
w = tf.Variable(4)
w.assign_sub(1)  #自减1
print(w)
#argmax返回张量沿指定维度最大值的索引 axis=0为纵向 1位横向
test = np.array([[1, 2, 3],[2, 3, 4],[5, 4, 3],[8, 7, 2]])
print(test)
print(tf.argmax(test,axis=0)) #返回每一列最大值的索引
print(tf.argmax(test,axis=1)) #返回每一行最大值的索引

