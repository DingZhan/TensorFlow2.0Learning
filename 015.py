#tf基础：用cat_vs_dogs数据集，但是貌似最后是Resource exhausted，是否不适合在CPU上运行这个sample？
#训练集下载地址 https://www.floydhub.com/fastai/datasets/cats-vs-dogs
import tensorflow as tf
import numpy as np
import os

num_epochs = 10
batch_size = 32
learning_rate = 0.001
data_dir = "D:/datasets/cats_vs_dogs"
train_cats_dir = data_dir+"/train/cats/"
train_dogs_dir = data_dir+"/train/dogs/"
test_cats_dir = data_dir+"/valid/cats/"
test_dogs_dir = data_dir+"/valid/dogs/"




def _decode_and_resize(filename, label):
    image_string = tf.io.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string)
    image_resized = tf.image.resize(image_decoded, [256, 256])/255.0
    return image_resized, label
    
    
if __name__=='__main__':
    #构建训练数据
    train_cat_filenames = tf.constant([train_cats_dir + filename for filename in os.listdir(train_cats_dir)])
    train_dog_filenames = tf.constant([train_dogs_dir + filename for filename in os.listdir(train_dogs_dir)])
    train_filenames = tf.concat([train_cat_filenames, train_dog_filenames], axis=-1)
    train_labels = tf.concat([
        tf.zeros(train_cat_filenames.shape, dtype=tf.int32),
        tf.ones(train_dog_filenames.shape, dtype=tf.int32)],
        axis=-1)
    
    train_dataset = tf.data.Dataset.from_tensor_slices((train_filenames, train_labels))
    train_dataset = train_dataset.map(_decode_and_resize)
    #取出前buffer_size个数据放入buffer，随机打乱
    train_dataset = train_dataset.shuffle(buffer_size=23000)
    train_dataset = train_dataset.batch(batch_size)
    
    #keras.Sequential训练方式
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(256, 256, 3)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 5, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(2, activation = 'softmax')
        ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss = tf.keras.losses.sparse_categorical_crossentropy,
        metrics =[tf.keras.metrics.sparse_categorical_accuracy]
        )
        
    model.fit(train_dataset, epochs=num_epochs)
    
    print("model.fit")
    #用以下代码进行测试
    #但貌似不管用啊
    test_cat_filenames = tf.constant([test_cats_dir + filename for filename in os.listdir(test_cats_dir)])
    test_dog_filenames = tf.constant([test_dogs_dir + filename for filename in os.listdir(test_dogs_dir)])
    test_filenames = tf.concat([test_cat_filenames, test_dog_filenames], axis=-1)
    test_labels = tf.concat([
        tf.zeros(test_cat_filenames.shape, dtype=tf.int32),
        tf.zeros(test_dog_filenames.shape, dtype=tf.int32)],
        axis=-1)

    test_dataset = tf.data.Dataset.from_tensor_slices((test_filenames, test_labels))
    test_dataset = test_dataset.map(_decode_and_resize)
    test_dataset = test_dataset.batch(batch_size)

    print(model.metrics_names)
    print(model.evaluate(test_dataset))    
        



