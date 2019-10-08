#tf基础: @tf.function修饰符。@tf.function带来一定的性能提升
import tensorflow as tf
import numpy as np
import time
#from zh.model.mnist.cnn import CNN
#from zh.model.utils import MNISTLoader


#载入MNIST训练样本库类
class MNISTLoader:
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
		
#定义卷积神经网络模型
class CNN(tf.keras.Model):
	def __init__(self):
		super().__init__()
		self.conv1 = tf.keras.layers.Conv2D(
			filters=32,
			kernel_size=[5,5],
			padding = 'same',
			activation=tf.nn.relu
		)
		self.pool1 = tf.keras.layers.MaxPool2D(pool_size=[2,2], strides=2)
		self.conv2 = tf.keras.layers.Conv2D(
			filters=64,
			kernel_size=[5,5],
			padding = 'same',
			activation = tf.nn.relu)
		self.pool2 = tf.keras.layers.MaxPool2D(pool_size=[2,2], strides=2)
		self.flatten = tf.keras.layers.Reshape(target_shape=(7*7*64,))
		self.dense1 = tf.keras.layers.Dense(units=1024, activation=tf.nn.relu)
		self.dense2 = tf.keras.layers.Dense(units=10)
	
	def call(self, inputs):
		x = self.conv1(inputs)
		x = self.pool1(x)
		x = self.conv2(x)
		x = self.pool2(x)
		x = self.flatten(x)
		x = self.dense1(x)
		x = self.dense2(x)
		output = tf.nn.softmax(x)
		return output
        
num_batches = 400
batch_size = 50
learning_rate = 0.001
data_loader = MNISTLoader()



#去掉@tf.function修饰符后，从31秒变成38秒，可见@tf.function对于模型训练提升是有作用的
@tf.function
def train_one_step(X, y):
    with tf.GradientTape() as tape:
        y_pred = model(X)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred = y_pred)
        loss = tf.reduce_mean(loss)
        #这里使用TensorFlow自己的print, @tf.function不支持Python内置的print
        tf.print("loss", loss)
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
    
if __name__ == '__main__':
    model = CNN()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    start_time = time.time()
    for batch_index in range(num_batches):
        x, y = data_loader.get_batch(batch_size)
        train_one_step(x, y)
    end_time = time.time()
    print(end_time - start_time)
    
    
    
         
       



