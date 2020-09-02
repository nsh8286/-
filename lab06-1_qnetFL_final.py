import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
#from tensorflow.keras import layers
import copy

env = gym.make('FrozenLake-v0')

#input and output size based on the Env
input_size = env.observation_space.n
output_size = env.action_space.n
learning_rate = 0.1

# choose actions
'''
inputs = tf.keras.Input(shape=(input_size))
W =  tf.Variable(tf.random.uniform(
    [input_size, output_size], 0, 1))#weight
Wbf = W.numpy()
print(W)
print(Wbf)
print(type(W))
print(type(Wbf))
Qpred = tf.matmul(inputs, W)
modelQpred = tf.keras.Model(inputs = inputs, outputs = Qpred)
'''
class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__(name = 'my_model')
        self.W = tf.Variable(tf.random.uniform(
                                [input_size, output_size], 0, 0.01))#weight
    def call(self,x):
        return tf.matmul(x,self.W)

modelQpred = MyModel()
Wbf = modelQpred.W.numpy() # W before fit 저장

def loss(predicted_y, desired_y):
    return tf.reduce_sum(tf.square(predicted_y-  desired_y))
loss_fn = lambda: loss(modelQpred(input), output)
var_list_fn = lambda: modelQpred.trainable_weights

'''
modelQpred.compile(loss = loss, optimizer = tf.keras.optimizers.SGD(lr=learning_rate))
modelQpred.summary()
'''
'''
def train(model, inputs, outputs, learning_rate):
    with tf.GradientTape() as t:
        current_loss = loss(model.predict(inputs), outputs)
    dW = t.gradient(current_loss, model.W)
    print(dW)
    model.W.assign_sub(learning_rate*dW)
'''
opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
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
        Qs = modelQpred.predict(one_hot(s))
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
        input = one_hot(s)
        output = Qs
        opt.minimize(loss_fn,var_list_fn)
        '''#for debug
        Waf = modelQpred.W.numpy()
        if (not np.array_equiv(Wbf, Waf)):
            print("W has changed!")
        if (done): # for watch the W change
            print(modelQpred.W)
            print('state : ', s)
            print('one hot : ', one_hot(s))
            print('Qs    : ', Qs)        
        '''
        if (done): print(i)


        rAll += reward
        s = s1
    rList.append(rAll)
print("Present of succesful episodes: " +
        str(sum(rList) / num_episodes))
plt.bar(range(len(rList)), rList, color="blue")
plt.show()