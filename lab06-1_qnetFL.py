import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

env = gym.make('FrozenLake-v0')

#input and output size based on the Env
input_size = env.observation_space.n
output_size = env.action_space.n
learning_rate = 0.1

tf.model = tf.keras.Sequential([
    
])