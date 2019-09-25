#使用tf.keras.Model
#本例中数学基础是f(A*W+b), f是激活函数，默认f(x)=x即不激活， A是输入矩阵， W是权重矩阵，b偏置向量bias
import tensorflow as tf
import numpy as np

xdata = np.array([2014,  2015,  2016,  2017,  2018], dtype = np.float32)
ydata = np.array([14500, 16000, 17200, 18900, 22100], dtype=np.float32)

#数据归一化
xdata = (xdata-xdata.min())/(xdata.max()-xdata.min())
ydata = (ydata-ydata.min())/(ydata.max()-ydata.min())

class LinearModel(tf.keras.Model):
	def __init__(self):
		super().__init__();
		#定义各层,这里使用全连接层，它相当于y = a*X+b中的线性层
		self.layer1 = tf.keras.layers.Dense(
			units=1,
			activation=None,
			kernel_initializer = tf.zeros_initializer(),
			bias_initializer=tf.zeros_initializer())

	#这里不是重载__call__而是 call
#	def __call__(self, input):
	def call(self, input):
		#对输入调用各层，然后输出
		output=self.layer1(input)
		return output
		

###ValueError: Input 0 of layer dense is incompatible with the layer: : expected min_ndim=2, found ndim=1. Full shape received: [5]
#X = tf.Variable(initial_value = xdata)
#Y = tf.Variable(initial_value = ydata)
#X = tf.constant(xdata)
#Y = tf.constant(ydata)

X = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
Y = tf.constant([[10.0], [20.0]])
limit = 1000

lm = LinearModel()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
for i in range(limit):
	with tf.GradientTape() as tape:
		y_pred = lm(X)
		#这里不是reduce_sum而是reduce_mean
		loss = 0.5*tf.reduce_mean(tf.square(Y-y_pred))
	grads = tape.gradient(loss, lm.variables)
	optimizer.apply_gradients(grads_and_vars = zip(grads, lm.variables))
	

print ("The final result ...")	
print (lm.variables)
	

