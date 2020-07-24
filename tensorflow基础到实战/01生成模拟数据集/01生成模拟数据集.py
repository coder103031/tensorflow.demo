#生成一个y≈2x的数据集
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#在内存中生成模拟数据
def GenerateData(batchsize = 100):             #自定义生成模拟数据函数
    train_x = np.linspace(-1, 1, batchsize)     #生成（-1,1）之间的100个浮点数
    train_y = 2 * train_x + np.random.randn(*train_x.shape) * 0.3   # y = 2x 加入了噪声
    yield train_x, train_y      #以生成器的方式返回

x_input = tf.placeholder("float", (None))  #定义两个占位符，用来接收参数
y_input = tf.placeholder("float", (None))
train_epoch = 200     #定义需要迭代次数
with tf.Session() as sess:              #建立会话
    for epoch in range(train_epoch):    #迭代数据集200遍
        for x, y in GenerateData():               #通过for循环打印所有的点
            xv, yv = sess.run([x_input,y_input], feed_dict= {x_input:x, y_input: y})
                                       #通过静态图的方式传入数据
            print (epoch, "| x.shape: ",np.shape(xv),"| x[:3}: ",xv[:3])
            print (epoch, "| y.shape: ", np.shape(yv), "| y[:3}: ", yv[:3])
                                #每次打印出迭代的次数|数据的形状|前三个元素
#显示模拟数据点，数据可视化
train_data = list(GenerateData())[0]             #获取数据
plt.plot(train_data[0],train_data[1], 'ro', label = 'Original data')    #生成图像
plt.legend()          #添加图例说明
plt.show()            #显示图像
