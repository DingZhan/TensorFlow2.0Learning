#tf求偏导入门，看不懂的就是tf.reduce_sum()这个作用是干什么,而且它的前面还要乘以0.5

import tensorflow as tf
print(tf.__version__)	#打印tf版本
X = tf.constant([[1,2],[3,4]], dtype=tf.float32)
y = tf.constant([[1],[2]], dtype=tf.float32)
w = tf.Variable(initial_value=[[1],[2]], dtype=tf.float32)
b = tf.Variable(initial_value=1, dtype=tf.float32)
with tf.GradientTape() as tape:
	L = 0.5*tf.reduce_sum(tf.square(tf.matmul(X, w)+b-y))		#reduce_sum是啥意思？

#tf.reduce_sum() 操作代表对输入张量的所有元素求和，输出一个形状为空的纯量张量

w_grad, b_grad = tape.gradient(L, [w, b])		#求L(w,b)关于w, b的偏导数

print(L.numpy())
print(w_grad.numpy())
print(b_grad.numpy())
