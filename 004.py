#tf线性回归， y = a*x+b, 求系数a, b
import tensorflow as tf
import numpy as np

print(tf.__version__)
print(np.__version__)

#x数据
x_raw = np.array([2013, 2014, 2015, 2016, 2017], dtype=np.float32)
#y数据
y_raw = np.array([12000, 14000, 15000, 16500, 17500], dtype=np.float32)

#数据归一化
xs =(x_raw-x_raw.min())/(x_raw.max()- x_raw.min())
ys =(y_raw-y_raw.min())/(y_raw.max()- y_raw.min())

print(xs)
print(ys)

#从numpy列表转换成tf的列表
x = tf.constant(xs) 
y = tf.constant(ys)
#设置两个tf的变量，作为带求解值
a = tf.Variable(initial_value=0, dtype=tf.float32)
b = tf.Variable(initial_value=0, dtype=tf.float32)
#构建待求解变量列表
ab = [a, b]

#构建一个梯度下降的优化器，学习效率为1.0e-3
optimizer = tf.keras.optimizers.SGD(learning_rate = 1.0e-3)
#最多迭代次数
limit = 1000
#开始迭代
for i in range(limit):
	#计算损失函数
	with tf.GradientTape() as tap:
		y2 = a*x+b;
		loss = 0.5*tf.reduce_sum(tf.square(y2-y))
	#计算梯度
	grads = tap.gradient(loss, ab)
	#根据梯度自动更新参数
	optimizer.apply_gradients(grads_and_vars=zip(grads, ab))
print("***********************")
print("a = ",a.numpy())
print("b = ",b.numpy())

#zip将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表
#如果 a = [1, 3, 5]， b = [2, 4, 6]，那么 zip(a, b) = [(1, 2), (3, 4), ..., (5, 6)]

