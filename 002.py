#ts变量、求导入门
import tensorflow as tf
print(tf.__version__) 				#打印tf版本号

x = tf.Variable(initial_value=3.)   #tf中的变量是默认能够被求导的
with tf.GradientTape() as tape:		#自动求导记录器
	y = tf.square(x)
#tf.assign(x, 4.0)   				#tf中的使用assign修改变量的值, 老的写法，
x.assign(4.0)						#tf中的使用assign修改变量的值，新的写法 https://tensorflow.google.cn/versions/r2.0/api_docs/python/tf/Variable
y_grad = tape.gradient(y, x)
print("***************")
print([y, y_grad])
print(y.numpy(), y_grad.numpy())