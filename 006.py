#tf基础：多层感知机(MLP)
import tensorflow as tf
import numpy as np

#载入MNIST训练样本库类
class MnistLoader:
	def __init__(self):
		#会自动下载MNIST训练样本库(C:/users/myname/.keras/datasets/mnist.npz
		mnist = tf.keras.datasets.mnist
		#读入下载样本库中的训练数据、训练标签、测试数据、测试标签
		(self.train_data, self.train_label),(self.test_data, self.test_label) = mnist.load_data()
		#在TensorFlow中，图像数据集的一种典型表示是 [图像数目，长，宽，色彩通道数] 的四维张量
		#因此需要使用 np.expand_dims() 函数为图像数据手动在最后添加一维通道。
		#但axis=-1表示啥意思？
		# [60000, 28, 28, 1]
		self.train_data = np.expand_dims(self.train_data.astype(np.float32)/255.0, axis=-1)
		# [10000, 28, 28, 1]
		self.test_data = np.expand_dims(self.test_data.astype(np.float32)/255.0, axis=-1)
		#看不懂是啥意思，但它提示是[60000]
		self.train_label = self.train_label.astype(np.int32)
		#看不懂是啥意思，但它提示是[10000]
		self.test_label = self.test_label.astype(np.int32)
		self.num_train_data = self.train_data.shape[0]
		self.num_test_data = self.test_data.shape[0]
		
	def get_batch(self, batch_size):
		#从数据集中随机取出batch_size个元素并返回
		index = np.random.randint(0, self.train_data.shape[0], batch_size)
		#但是train_data和train_label的切片貌似不对齐啊,一个是index-end, 另一个是start-index
		return self.train_data[index, :], self.train_label[index]
		
class MLP(tf.keras.Model):
	def __init__(self):
		super().__init__()
		self.flatten = tf.keras.layers.Flatten()
		self.dense1 = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)
		self.dense2 = tf.keras.layers.Dense(units=10)
		
	def call(self, input):	#[batch_size, 28, 28, 1]
		#[batch_size, 784]
		x = self.flatten(input)	
		#[batch_size, 100]
		x = self.dense1(x)
		#[batch_size, 10]
		x = self.dense2(x)
		#使用归一化指数函数，对原始输出进行归一化，使得10维(数字0-9)向量的每个元素均在[0,1]之间,并且向量之和为1
		#softmax 函数能够凸显原始向量中最大的值，并抑制远低于最大值的其他分量
		output = tf.nn.softmax(x)
		return output
		
		

#模型的训练	tf.keras.losses 和 tf.keras.optimizer	
		
num_epochs = 5	
batch_size = 50
#learning rate
lrate = 0.001 

model = MLP()
mnist = MnistLoader()
optimizer = tf.keras.optimizers.Adam(learning_rate = lrate)

num_batches = int(mnist.num_train_data//batch_size*num_epochs)
for batch_index in range(num_batches):
	x, y = mnist.get_batch(batch_size)
	with tf.GradientTape() as tape:
		y_pred = model(x)
		#这里没有显式地写出一个损失函数，而是使用了sparse_categorical_crossentropy （交叉熵）函数		
		#将模型的预测值 y_pred 与真实的标签值 y 作为函数参数传入，由 Keras 帮助我们计算损失函数的值
		loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred = y_pred)
		loss = tf.reduce_mean(loss)
		print("batch %d: loss %f"%(batch_index, loss.numpy()))
	grads = tape.gradient(loss, model.variables)
	optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
	
	
#模型的评估 tf.keras.metrics
sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
num_batches = int(mnist.num_test_data//batch_size)
for batch_index in range(num_batches):
	startID, endID = batch_index*batch_size, (batch_index+1)*batch_size
	y_pred = model.predict(mnist.test_data[startID:endID])
	sparse_categorical_accuracy.update_state(y_true=mnist.test_label[startID:endID], y_pred=y_pred)
print("test accuracy: %f"%sparse_categorical_accuracy.result())
