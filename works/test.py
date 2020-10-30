import torch
from actornetwork import MuNet
from criticnetwork import QNet
from test2 import Test2Class
from gym_torcs import TorcsEnv
from replaybuffer import ReplayBuffer
import numpy as np
import timeit
from ou import OrnsteinUhlenbeckNoise
# # --nn input output test--
# state_dim = 8
# action_dim = 2
# actor = MuNet(state_dim=state_dim)
# critic = QNet(state_dim=state_dim, action_dim = action_dim)
# steer_noise = OrnsteinUhlenbeckNoise(mu = np.zeros(1))
# accel_noise = OrnsteinUhlenbeckNoise(mu = np.zeros(1))
# a_t = np.zeros([1,action_dim])

# state = torch.rand(1,8)
# print ("state: ",state)
# action_output_original = actor(state)
# print ("action_output_original: ",action_output_original)

# a_t[0][0] = action_output_original.detach().numpy()[0][0] + steer_noise()
# a_t[0][1] = action_output_original.detach().numpy()[0][1] + accel_noise()
# print("steer noise: ",steer_noise)
# print("a_t: ", a_t)
# action_output = torch.tensor(a_t,dtype=torch.float)
# print("action_output: ", action_output)
# critic_output = critic(state,action_output)
# print ("critic_output: ",critic_output)

# # --import class parameter test--
# num = 100
# instance = Test2Class()
# print(instance)
# print(instance())

# # --gym_torcs and replaybuffer test--
# env = TorcsEnv(vision=False, throttle=True, gear_change=False)
# memory = ReplayBuffer()
# max_step = 500
# ob = env.reset()
# s_t = np.hstack((ob.angle, ob.track,ob.trackPos,ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
# a = [[0,0]]
# t_start = timeit.default_timer()
# for i in range(max_step):
#     ob,r_t,done,info = env.step(a[0])
#     if done:
#         break
#     s_t1 = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
#     memory.put((s_t,a[0],r_t,s_t1,done))
#     s_t = s_t1
# t_end = timeit.default_timer()
# s_done = s_t
# print ('done?: ',s_done)
# print('{}steps, {} time spent'.format(i,t_end-t_start))
# env.end()
# s,a,r,sp,d = memory.sample(1)
# print('s: ',s)
# print('a: ',a)
# print('r: ',r)
# print('sp: ', sp)
# print('d: ',d)

# --noise 테스트합니다.--
noise = OrnsteinUhlenbeckNoise(mu = np.zeros(1),theta=1,sigma = 0.5)
for i in range(300):
    noise()
    print(noise)
