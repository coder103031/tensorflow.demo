import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # 只显示 warning 和 Error
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0      #对输入网络的数据进行归一化，使得0-255的灰度值变为0-1之间

model = tf.keras.models.Sequential([          #使用Sequential搭建神经网络 将输入特征拉直一维数组即784个数值
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),       #定义第一层有128个神经网络 以relu为激活函数
    tf.keras.layers.Dense(10, activation='softmax')       #定义第二层有10个神经元 用softmax函数 输出符合概率分布
])

model.compile(optimizer='adam',     #compile配制训练方法，优化器选择adam，
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),  #损失函数选择SparseCategoricalCrossentropy，输出层是概率分布，而非直接输出，所以这里是False
              metrics=['sparse_categorical_accuracy'])      #数据集中的标签是数值 神经网络输出y是概率分布 所以这里选择sparse_categorical_accuracy

model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), validation_freq=1)
#fit中执行训练过程，训练集输入特征，每次喂入网络32组数据，数据集迭代5次，每迭代一次训练集，进行一次测试集评测
model.summary()      #打印网络结构和参数统计