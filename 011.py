#tf基础：Deep Learning OpenAI 游戏 GYM环境
import tensorflow as tf
import numpy as np
import gym
import random
from collections import deque

#最大训练次数（重启游戏次数,或最多多少局游戏),缺省1000
num_episodes = 20
num_exploration_episodes = 100
#每局游戏的最多做多少次动作（中途可能会死掉)
max_len_episode = 1000
batch_size = 32
learning_rate = 0.001
gamma = 1.
initial_epsilon = 1.
final_epsilon = 0.01

#Q-network用于拟合Q函数，输入状态state，输出各个action下的Q-value（CartPole中只有两个action）
class QNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(units=24, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=24, activation=tf.nn.relu)
        #结果为二维向量
        self.dense3 = tf.keras.layers.Dense(units=2)
        
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return x
    
    def predict(self, inputs):
        q_values = self(inputs)
        return tf.argmax(q_values, axis=-1)

if __name__=="__main__":
    env = gym.make("CartPole-v1")
    model = QNetwork()
    optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
    replay_buffer = deque(maxlen=10000)
    epsilon = initial_epsilon
    for episode_id in range(num_episodes):
        state = env.reset()
        epsilon = max(
            initial_epsilon*(num_exploration_episodes-episode_id)/num_exploration_episodes,
            final_epsilon)
        for t in range(max_len_episode):
            env.render()
            #epsilon-greedy 探索策略
            if random.random()<epsilon:
                action = env.action_space.sample()
            else:
                action = model.predict(
                    tf.constant(np.expand_dims(state, axis=0), dtype=tf.float32)).numpy()
                action = action[0]

            # 让环境执行动作，获得执行完动作的下一个状态，动作的奖励，游戏是否已结束以及额外信息
            next_state, reward, done, info = env.step(action)
            # 如果游戏Game Over，给予大的负奖励
            reward = -10. if done else reward
            # 将(state, action, reward, next_state)的四元组（外加done标签表示是否结束）放入经验重放池
            replay_buffer.append((state, action, reward, next_state, 1 if done else 0))
            # 更新当前state
            state = next_state

            if done:                                    # 游戏结束则退出本轮循环，进行下一个episode
                print("episode %d, epsilon %f, score %d" % (episode_id, epsilon, t))
                break

            if len(replay_buffer) >= batch_size:
                # 从经验回放池中随机取一个批次的四元组，并分别转换为NumPy数组
                batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(
                    *random.sample(replay_buffer, batch_size))
                batch_state, batch_reward, batch_next_state, batch_done = \
                    [np.array(a, dtype=np.float32) for a in [batch_state, batch_reward, batch_next_state, batch_done]]
                batch_action = np.array(batch_action, dtype=np.int32)

                q_value = model(tf.constant(batch_next_state, dtype=tf.float32))
                y = batch_reward + (gamma * tf.reduce_max(q_value, axis=1)) * (1 - batch_done)  # 按照论文计算y值
                with tf.GradientTape() as tape:
                    loss = tf.keras.losses.mean_squared_error(  # 最小化y和Q-value的距离
                        y_true=y,
                        y_pred=tf.reduce_sum(model(tf.constant(batch_state)) * tf.one_hot(batch_action, depth=2), axis=1)
                    )
                grads = tape.gradient(loss, model.variables)
                optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))       # 计算梯度并更新参数
               
print("Start auto gaming...")
#实列化一个游戏环境，参数为游戏名称
env = gym.make('CartPole-v1')
#初始化环境，得到初始状态
state = env.reset()

#model = QNetwork()
#游戏运行
while True:
    #渲染游戏
    env.render()
    #通过训练好的模型，根据当前状态做出动作
    #这句话会引起异常，根据显示貌似是state矩阵和dense1所需要的矩阵尺寸不同引起的
    action = model.predict(state)
    #让游戏执行该动作
    next_state, reward, done, info = env.step(action)
    #如果游戏结束则退出循环
    if done:
        break;