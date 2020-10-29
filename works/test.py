import torch
from actornetwork import MuNet
from criticnetwork import QNet
from test2 import Test2Class
from gym_torcs import TorcsEnv
from replaybuffer import ReplayBuffer
import numpy as np
# # --nn input output test--
# state_dim = 8
# action_dim = 2
# actor = MuNet(state_dim=state_dim)
# critic = QNet(state_dim=state_dim, action_dim = action_dim)

# state = torch.rand(4,8)
# print ("state: ",state)
# action_output = actor(state)
# print ("action_output: ",action_output)
# critic_output = critic(state,action_output)
# print ("critic_output: ",critic_output)

# # --import class parameter test--
# num = 100
# instance = Test2Class()

# print(instance())

# --gym_torcs and replaybuffer test--
env = TorcsEnv(vision=False, throttle=True, gear_change=False)
memory = ReplayBuffer()
max_step = 500
ob = env.reset()
s_t = np.hstack((ob.angle, ob.track,ob.trackPos,ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
a = [[0.1,0.8]]
for i in range(max_step):
    ob,r_t,done,info = env.step(a[0])
    if done:
        break
    s_t1 = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
    memory.put((s_t,a[0],r_t,s_t1,done))
env.end()
s,a,r,sp,d = memory.sample(1)
print('s: ',s)
print('a: ',a)
print('r: ',r)
print('sp: ', sp)
print('d: ',d)



