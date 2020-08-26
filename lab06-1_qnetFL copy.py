import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

env = gym.make('FrozenLake-v0')

#input and output size based on the Env
input_size = env.observation_space.n
output_size = env.action_space.n
learning_rate = 0.1

# choose actions
inputs = tf.keras.Input(shape=(input_size))
W =  tf.Variable(tf.random.uniform(
    [input_size, output_size], 0, 0.01))#weight
Qpred = tf.matmul(inputs, W)
modelQpred = tf.keras.Model(inputs = inputs, outputs = Qpred)

modelQpred.compile(loss = 'mse',optimizer = tf.keras.optimizers.SGD(lr=learning_rate))
modelQpred.summary()

# Set Q-learning related parameters
dis =.99
num_episodes = 2000

#Create lists to contain total rewards and steps per episode
rList =[]

def one_hot(x):
    return np.identity(16)[x:x+1] # 모양 조심

for i in range(num_episodes):
    #reset environment and get first new observation
    s = env.reset()
    e = 1./((i/50)+10)
    rAll = 0
    done = False
    #local loss는 따로 안구하기로 함

    #the Q-network training
    while not done:
        # Choose an action by greedily (with e chance of random action)
        # from the Q-network
        tmp = one_hot(s)
        Qs = modelQpred.predict(tmp)
        if np.random.rand(1) < e :
            a = env.action_space.sample()
        else:
            a = np.argmax(Qs)
        
        #get new state and reward from environment
        s1, reward, done, _ = env.step(a)
        if done:
            #update Q, and no Qs+1, since it's a terminal state
            Qs[0,a] = reward
        else:
            # Obtain the Q_s1 values by feeding the new state through our
            # network
            Qs1 = modelQpred.predict(one_hot(s1))
            #update Q
            Qs[0,a] = reward + dis*np.max(Qs1)

        #train our network using target Y and predicte Q values
        modelQpred.fit(one_hot(s),Qs, verbose =0)

        rAll += reward
        s = s1
    rList.append(rAll)
print("Present of succesful episodes: " +
        str(sum(rList) / num_episodes)+ "%")
plt.bar(range(len(rList)), rList, color="blue")
plt.show()


