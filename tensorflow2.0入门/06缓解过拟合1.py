import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # 只显示 warning 和 Error

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#创建需要拟合的数据 先看输出结果，我们创建了x1,x2作为输入特征，y_c是对应的标签。y_c值为1的点为红色
#所有值为0的话标为蓝色。然神经网络画出一条线来区分红色和蓝色
# BATCH_SIZE=30 	#一次喂入神经网络的30组数据
seed=2
#基于seed产生随机数
rdm=np.random.RandomState(seed)
#随机数返回300行2列的矩阵，表示300组坐标点（x0,x1）作为输入数据集
X=rdm.randn(300,2)
#从X这个300行2列的矩阵中取出1行，判断如果2个坐标的平方和小于2，给Y赋值为1，其他赋值为0
#作为数据集的正确答案（标签）
Y_=[int(x0*x0+x1*x1<2) for (x0,x1) in X ]
#遍历Y_中的每个元素，1赋值为red,2赋值为blue,这样可视化显示时人可以直观的区分
Y_c=[['red' if y else 'blue'] for y in Y_]
#对数据集X和标签Y进行shape整理，第一元素为-1表示，随第二个参数计算得到，第二个元素表示多少列，把X整理为n行2列，把Y整理为n行1列
# X=np.vstack(X).reshape(-1,2)
Y=np.vstack(Y_).reshape(-1,1)
print(X)
print(Y)
print(Y_c)
#用plt.scatter画出数据集X和各行中的第0列元素和第1列元素的点即各行的（x0,x1)，用各行Y_c对应的值表示颜色（c是color的缩写）
plt.scatter(X[:,0],X[:,1],c=np.squeeze(Y_c))
plt.show()
