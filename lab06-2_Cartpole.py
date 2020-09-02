import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

env = gym.make('CartPole-v0')

input_size = env.observation_space.shape[0]
output_size = env.action_space.n
learning_rate = 0.1

#make model for Qpred
class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__(name = 'my_model')
        initializer = tf.initializers.GlorotUniform()
        self.W = tf.Variable(initializer(
            shape = (input_size, output_size)))#weight initializing?
    def call(self,x):
        return tf.matmul(x,self.W)

modelQpred = MyModel()
#square
def loss(predicted_y, desired_y):
    return tf.reduce_sum(tf.square(predicted_y - desired_y))
#ready for opt.minimize
loss_fn = lambda: loss(modelQpred(input),output)
var_list_fn = lambda: modelQpred.trainable_weights
opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)

# Set Q-learning related parameters
dis =.9
num_episodes = 2000

#Create lists to contain total rewards and steps per episode
rList =[]

for i in range(num_episodes):
    #reset environment and get first new observation
    s = env.reset()
    e = 1./((i/10)+1)
    rAll = 0
    step_count =0
    done = False

    while not done:
        step_count +=1
        x = np.reshape(s, [1,input_size]) #lab06-1에서 one_hot 리턴값과 동일모양
        Qs = modelQpred.predict(x)
        if np.random.rand(1)< e:
            a = env.action_space.sample()
        else:
            a = np.argmax(Qs)
        
        #get new state and reward from environment
        s1, reward, done, _ = env.step(a)
        if done:
            #update Q, and no Qs+1, since it's a terminal state
            Qs[0,a] = -100 # penalty for failure
        else:
            # Obtain the Q_s1 values by feeding the new state through our
            # network
            x1 = np.reshape(s1, [1, input_size])
            Qs1 = modelQpred.predict(x1)
            #update Q
            Qs[0,a] = reward + dis*np.max(Qs1)
        
        #train our network using target Y and predicte Q values
        input = x
        output = Qs
        opt.minimize(loss_fn,var_list_fn)
        s = s1

    rList.append(step_count)
    print("Episode: {} steps: {}".format(i, step_count))
    # If last 10's avg step are 500, it's good enough
    if len(rList) > 10 and np.mean(rList[-10:]) >500:
        break

#see our trained network in action
observation = env.reset()
reward_sum = 0
while True:
    env.render()

    x = np.reshape(observation, [1, input_size])
    Qs = modelQpred.predict(x)
    a = np.argmax(Qs)

    observation, reward, done, _ = env.step(a)
    reward_sum += reward
    if done:
        print("Total score: {}".format(reward_sum))
        break