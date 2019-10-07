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

singleShow = False

if singleShow:
    #如何不关闭窗口切换到下一副训练集合图片,只能通过关闭方式吗?
    for image, label in mnist_dataset:
        plot.title(label.numpy())
        plot.imshow(image.numpy()[:,:,0])
        plot.show()
        
else:
    row = 4
    col = 6
    mnist_dataset = mnist_dataset.batch(row*col)
    #打散数据集
    #mnist_dataset = mnist_dataset.shuffle(buffer_size=10000).batch(4)
    for images, labels in mnist_dataset:    # image: [4, 28, 28, 1], labels: [4]
        k = 0
        fig, axs = plot.subplots(row, col)
        for i in range(row):
            for j in range(col):
                axs[i][j].set_title(labels.numpy()[k])
                axs[i][j].imshow(images.numpy()[k, :, :, 0])
                k = k+1
        plot.show()