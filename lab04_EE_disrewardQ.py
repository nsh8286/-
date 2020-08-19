import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register
import random as pr

def rargmax(vector):
    # choose maximum - random indice
    m = np.amax(vector)
    indices = np.nonzero(vector==m)[0]
    return pr.choice(indices)

register(
    id = 'FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False}
)
env = gym.make('FrozenLake-v3')

#initialize table with all zeros
Q = np.zeros([env.observation_space.n, env.action_space.n])
#discount factor
dis = .99
#set learning parameters
num_episodes = 2000

#create lists to contain totla rewards and steps per episode
rList =[]
for i in range(num_episodes):
    #reset environment and get first new observation
    state = env.reset()
    rAll =0
    done = False

    #the Q-table learning algorithm
    while not done:
        #choose an action greedily
        noise = np.random.randn(env.action_space.n)/(i+1)*50
        action = rargmax(Q[state, :]+noise)

        #get new state and reward from environment
        new_state, reward, done, _ = env.step(action)

        #update Q-table with new knowledge using learning rate
        Q[state, action] = reward + dis*np.max(Q[new_state,:])

        rAll += reward
        state = new_state
    
    rList.append(rAll)


print("Success rate: " + str(sum(rList)/num_episodes))
print("Final Q-Table Values")
print ("LEFT DOWN RIGHT UP")
print(Q)
plt.bar(range(len(rList)), rList, color = "blue")
plt.show()