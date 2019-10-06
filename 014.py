#tf基础：用tf.data.Dataset.from_tensor_slices自建数据集，用matplotlib绘制MNIST 数据集
import tensorflow as tf
import matplotlib.pyplot as plot
import numpy as np

(train_data, train_label),(_,_) = tf.keras.datasets.mnist.load_data()
#[60000, 28, 28, 1]
#忘记了为什么要expand_dims?
train_data = np.expand_dims(train_data.astype(np.float32)/255.0, axis=-1)
#构建数据集
mnist_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_label))

#如何切换到下一副训练集合图片?
for image, label in mnist_dataset:
    plot.title(label.numpy())
    plot.imshow(image.numpy()[:,:,0])
    plot.show()
    
