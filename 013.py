#tf基础：训练过程可视化
#要对训练过程可视化时，进入 TensorFlow 的 conda 环境），运行 tensorboard --logdir=./tensorboard
#然后使用浏览器访问命令行程序所输出的网址（一般是 http:// 计算机名称：6006），即可访问 TensorBoard 的可视界面

import tensorflow as tf
import numpy as np

modelsavedpath = './savedmodel'

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
#        self.train_finished = 0
    

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

def LoadTrainedModel(nc, bs, sl):
    try:
        print("LoadTrainedModel  1")
        model = RNN(num_chars= nc, batch_size = bs, seq_length=sl)
        print("LoadTrainedModel  2")
        optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
        checkpoint = tf.train.Checkpoint(myOptimizer=optimizer, myRNNTextModel=model)
        print("LoadTrainedModel  3")
        status = checkpoint.restore(tf.train.latest_checkpoint(modelsavedpath))
#        if mode.train_finished==0:
#            return None
        print("LoadTrainedModel  4")
        #assert_consumed一加上去就有异常，不知道原因
        status.assert_consumed()
        print("LoadTrainedModel  5")
        return model
    except:
        return None

#定义模型超参数
#这里为了调试，我设置成20，最终还是要还原成1000
num_batches = 100
seq_length = 40
batch_size = 50
learning_rate = 0.001    
#定义一个数据载入器
data_loader = DataLoader()
#尝试从文件中载入之前可能保存的模型
model = LoadTrainedModel(len(data_loader.chars), batch_size, seq_length)
#如果之前没有保存过模型，则开始训练
if model==None:
    #定义一个RNN模型
    model = RNN(num_chars= len(data_loader.chars), batch_size = batch_size, seq_length=seq_length)
    #定义一个优化器
    optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)

    checkpoint = tf.train.Checkpoint(myOptimizer=optimizer, myRNNTextModel=model)
    manager = tf.train.CheckpointManager(checkpoint, directory=modelsavedpath, max_to_keep = 3)
    '''
    try:
        #尝试从文件中载入之前训练的模型
        model = tf.saved_model.load("rnn.model")
    except:
    '''
    #训练记录的目录
    summary_writer = tf.summary.create_file_writer('./tensorboard')
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
        
        if batch_index%100==0:
            path = manager.save(checkpoint_number=batch_index)
            print("current trained model saved to %s"%path)
            
        #使用记录器记录训练数据
        with summary_writer.as_default():
            tf.summary.scalar('loss', loss, step=batch_index)
            #tf.summary.scalar('myscalar', my_scalar, step=batch_index)
            
        
            
    #将该模型存入rnn.model文件中，从而下次可以直接载入
    #tf.saved_model.save(model, "rnn.model")
    
    #model train finished
#    model.train_finished = 1;
    path = manager.save(checkpoint_number=batch_index)
    print("finished trained model saved to %s"%path)
    


X_, _ = data_loader.get_batch(seq_length, 1)
for diversity in [0.2, 0.5, 1.0, 1.2]:
    X = X_
    print("diversity %f:" % diversity)
    for t in range(400):
        y_pred = model.predict(X, diversity)
        print(data_loader.indices_char[y_pred[0]], end='', flush=True)
        X = np.concatenate([X[:, 1:], np.expand_dims(y_pred, axis=1)], axis=-1)
    print("\n")
    
        
