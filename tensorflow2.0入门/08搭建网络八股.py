#用Tensorflow API：tf.keras搭建网络八股
# 六步法
# import    相关模块
# train, test          告知要喂入网络的训练集和测试集是什么
# model = tf.keras.models.Sequential   在Sequential中搭建网络结构 逐层描述网络结构 相当走一遍前向传播
# model.compile          配制训练方法 告知训练时用那种优化器 那个损失函数选择那种评测指标
# model.fit            告知训练集和测试集的输入特征和标签 告知每个batch是多少 告知要迭代多少次数据集
# model.summary          打印网络的结构和参数统计

# model = tf.keras.models.Sequential([网络结构])       #描述各层网络
# 网络结构有 拉直层： tf.keras.layers.Flatten()
# 全连接层：tf.keras.layers.Dense(神经元个数，activation = "激活函数"，kernel_regularizer = 哪种正则化)
# 卷积层，
# LSTM层

import tensorflow as tf
from matplotlib import pyplot as plt

mnist = tf.keras.datasets.mnist     #mnist共有7w张图 都是28*28像素点大小的手写数字 其中6w张训练1w张测试，
# 可以直接使用load_data（）直接读取mnist数据
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 可视化训练集输入特征的第一个元素
plt.imshow(x_train[0], cmap='gray')  # 绘制灰度图 将训练集x[0]的数据可视化输出
plt.show()

# 打印出训练集输入特征的第一个元素
print("x_train[0]:\n", x_train[0])    #打印第一个函数的输入特征 是个28*28的二维数组
# 打印出训练集标签的第一个元素
print("y_train[0]:\n", y_train[0])          #打印标签

# 打印出整个训练集输入特征形状
print("x_train.shape:\n", x_train.shape)
# 打印出整个训练集标签的形状
print("y_train.shape:\n", y_train.shape)
# 打印出整个测试集输入特征的形状
print("x_test.shape:\n", x_test.shape)
# 打印出整个测试集标签的形状
print("y_test.shape:\n", y_test.shape)
