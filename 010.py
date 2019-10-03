#tf基础：RNN 文本预测
import tensorflow as tf
import numpy as np
import pickle

class DataLoader:
	def __init__(self):
		path = tf.keras.utils.get_file("nietzsche.txt",
		origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")
		with open(path, encoding='utf-8') as f:
			self.raw_text = f.read().lower()
		self.chars = sorted(list(set(self.raw_text)))
		self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
		self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
		self.text = [self.char_indices[c] for c in self.raw_text]
		
	def get_batch(self, seq_length, batch_size):
		seq = []
		next_char = []
		for i in range(batch_size):
			index = np.random.randint(0, len(self.text) - seq_length)
			seq.append(self.text[index:index+seq_length])
			next_char.append(self.text[index+seq_length])
		return np.array(seq), np.array(next_char)
		
class RNN(tf.keras.Model):
	def __init__(self, num_chars, batch_size, seq_length):
		super().__init__()
		self.num_chars = num_chars
		self.seq_length = seq_length
		self.batch_size = batch_size
		self.cell = tf.keras.layers.LSTMCell(units=256)
		self.dense = tf.keras.layers.Dense(units = self.num_chars)
	

	#为了支持Keras 模型的导入和导出，这里需要加入@tf.function修饰符
	@tf.function
	def call(self, inputs, from_logits = False):
		#首先对序列进行One Hot操作，即将序列中的每个字符的编码i均变换为一个num_char维向量
		inputs = tf.one_hot(inputs, depth=self.num_chars)
		state = self.cell.get_initial_state(batch_size=self.batch_size, dtype=tf.float32)
		for t in range(self.seq_length):
			output, state = self.cell(inputs[:, t, :], state)
		logits = self.dense(output)
		if from_logits:
			return logits
		else:
			return tf.nn.softmax(logits)
		
	#之前一直使用tf.argmax()，将对应概率最大的值作为预测值。但对文本生成，这样的预测方式过于绝对，会使得生成的文本失去丰富性。
	#这里使用 np.random.choice() 函数按照生成的概率分布取样。同时加入temperature参数控制分布的形状
	
	#为了支持Keras 模型的导入和导出，这里需要加入@tf.function修饰符
	#但貌似出了这个错误：tensorflow.python.framework.errors_impl.OperatorNotAllowedInGraphError: iterating over `tf.Tensor` is not allowed: AutoGraph did not convert this function. Try decorating it directly with @tf.function	
	#@tf.function
	def predict(self, inputs, temperature=1.):
		#import numpy as np
		batch_size, _ = tf.shape(inputs)
		logits = self(inputs, from_logits = True)
		prob = tf.nn.softmax(logits/temperature).numpy()
		return np.array([np.random.choice(self.num_chars, p=prob[i, :])
			for i in range(batch_size.numpy())])

		
#定义模型超参数
#这里为了调试，我设置成20，最终还是要还原成1000
num_batches = 20
seq_length = 40
batch_size = 50
learning_rate = 0.001

#定义一个数据载入器
data_loader = DataLoader()
#定义一个RNN模型
model = RNN(num_chars= len(data_loader.chars), batch_size = batch_size, seq_length=seq_length)
#定义一个优化器
optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)

'''
try:
	#尝试从文件中载入之前训练的模型
	model = tf.saved_model.load("rnn.model")
except:
'''
#开始学习，学习1000遍
for batch_index in range(num_batches):
	#从DataLoader中随机取一批训练数据
	x, y = data_loader.get_batch(seq_length, batch_size)
	with tf.GradientTape() as tape:
		#将这批数据送入模型，计算出模型的预测值
		y_pred = model(x)
		#将模型预测值与真实值进行比较，计算损失函数(loss)
		loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred = y_pred)
		loss = tf.reduce_mean(loss)
		print('batch %d: loss %f'%(batch_index, loss.numpy()))
	#计算损失函数关于模型变量的导数
	grads = tape.gradient(loss, model.variables)
	#使用优化器更新模型参数
	optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
#将该模型存入rnn.model文件中，从而下次可以直接载入
tf.saved_model.save(model, "rnn.model")
'''
f = open("rnn.dat", "wb")
pickle.dump(model, f)
f.close()
'''


X_, _ = data_loader.get_batch(seq_length, 1)
for diversity in [0.2, 0.5, 1.0, 1.2]:
	X = X_
	print("diversity %f:" % diversity)
	for t in range(400):
		y_pred = model.predict(X, diversity)
		print(data_loader.indices_char[y_pred[0]], end='', flush=True)
		X = np.concatenate([X[:, 1:], np.expand_dims(y_pred, axis=1)], axis=-1)
	print("\n")
	
		
