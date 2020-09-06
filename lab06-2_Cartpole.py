import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import timeit

env = gym.make('CartPole-v0')

input_size = env.observation_space.shape[0]
output_size = env.action_space.n
learning_rate = 0.1

timestart = timeit.default_timer()
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
num_episodes = 15

#Create lists to contain total rewards and steps per episode
rList =[]
sumtime12 = 0
sumtime23 = 0

for i in range(num_episodes):
    #reset environment and get first new observation
    s = env.reset()
    e = 1./((i/10)+1)
    rAll = 0
    step_count =0
    done = False
    time1 = []
    time2 = []
    time3 = []
    time12 = []
    time23 = []

    while not done:
        time1.append(timeit.default_timer())#timer1
        step_count +=1
        x = np.reshape(s, [1,input_size]) #lab06-1에서 one_hot 리턴값과 동일모양
        time2.append(timeit.default_timer())#timer2
        Qstensor = modelQpred(x,training=False)
        Qs = Qstensor.numpy()
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
            Qs1 = modelQpred(x1, training= False)
            #update Q
            Qs[0,a] = reward + dis*np.max(Qs1)
        
        #train our network using target Y and predicte Q values
        input = x
        output = Qs
        opt.minimize(loss_fn,var_list_fn)
        s = s1
        time3.append(timeit.default_timer())#timer3

    #get timegaps
    time12 = np.array(time2)-np.array(time1)
    time23 = np.array(time3)-np.array(time2)
    sumtime12+=time12.sum()
    sumtime23+=time23.sum()

    rList.append(step_count)
    print("Episode: {} steps: {}".format(i, step_count))
    # If last 10's avg step are 500, it's good enough
    if len(rList) > 10 and np.mean(rList[-10:]) >500:
        break
timeend = timeit.default_timer()
print("time12 : ", sumtime12)
print("time23 : ", sumtime23)
print("whole time: ", timeend - timestart)

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