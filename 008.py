#tf基础：利用已有（第一次要下载）的训练模型MobileNetV2，识别花类型
'''
ModuleNotFoundError: No module named 'tensorflow_datasets' 问题
需要到anaconda prompt里，激活tensorflow后，pip install tensorflow-datasets
'''

import tensorflow as tf
import tensorflow_datasets as tfds

num_batches = 1000
batch_size = 50
learning_rate = 0.001

dataset = tfds.load("tf_flowers", split=tfds.Split.TRAIN, as_supervised=True)
dataset = dataset.map(lambda img, label:(tf.image.resize(img, [224,224])/255.0, label)).shuffle(1024).batch(32)
#使用已有的模型MobileNetV2
model = tf.keras.applications.MobileNetV2(weights = None, classes=5)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
for images, labels in dataset:
	with tf.GradientTape() as tape:
		label_pre = model(images)
		loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=labels, y_pred=label_pre)
		loss = tf.reduce_mean(loss)
		print("loss %f"%loss.numpy())
	grads = tape.gradient(loss, model.trainable_variables)
	optimizer.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))
	
		
