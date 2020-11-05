import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import datetime
import os
from torch.utils.tensorboard import SummaryWriter
from actornetwork import MuNet
from criticnetwork import QNet
from gym_torcs import TorcsEnv
from replaybuffer import ReplayBuffer
from ou import OrnsteinUhlenbeckNoise as OUN

state_dim = 29
action_dim = 2
max_episode = 1
max_step = 2000
log_timer = 0

EXPLORE      = 3000
lr_mu        = 0.0001
lr_q         = 0.001
gamma        = 0.99
batch_size   = 32
tau          = 0.001

def main():    
    global log_timer
    env = TorcsEnv(vision=False, throttle=True, gear_change=False)
    memory = ReplayBuffer()
    epsilon = 1
    train_indicator = True

    q,q_target = QNet(state_dim,action_dim),QNet(state_dim,action_dim)
    q_target.load_state_dict(q.state_dict())
    mu, mu_target = MuNet(state_dim), MuNet(state_dim)
    mu_target.load_state_dict(mu.state_dict())
    steer_noise = OUN(np.zeros(1),theta = 0.6)
    accel_noise = OUN(np.zeros(1),theta = 0.6)
    mu_optimizer = optim.Adam(mu.parameters(), lr=lr_mu)
    q_optimizer  = optim.Adam(q.parameters(), lr=lr_q)

    #tensorboard writer
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("logs", "ddpg_torch", current_time)
    writer = SummaryWriter(log_dir)
    samplestate = torch.rand(1,29)
    sampleaction = torch.rand(1,2)

    writer.add_graph(mu,samplestate)
    writer.add_graph(q,(samplestate,sampleaction))
    writer.close


    for n_epi in range(max_episode):
        print("Episode : " + str(n_epi) + " Replay Buffer " + str(memory.size()))
        if np.mod(n_epi, 3) == 0:
            ob = env.reset(relaunch=True)   #relaunch TORCS every 3 episode because of the memory leak error
        else:
            ob = env.reset()
        a_t = np.zeros([1,action_dim])
        s_t = np.hstack((ob.angle, ob.track,ob.trackPos,ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
        score = 0
        #t_start = timeit.default_timer()
        for n_step in range(max_step):
            epsilon -= 1.0/EXPLORE
            a_origin = mu(torch.from_numpy(s_t.reshape(1,-1)).float())
            if train_indicator == True:#add noise for train
                a_s = a_origin.detach().numpy()[0][0] + epsilon*steer_noise()
                a_t[0][0] = np.clip(a_s,-1,1) # fit in steer arange
                a_a = a_origin.detach().numpy()[0][1] + epsilon*accel_noise()
                a_t[0][1] = np.clip(a_a,0,1) # fit in accel arange
            else:
                a_t = a_origin.detatch().numpy()
            ob,r_t,done,info = env.step(a_t[0])
            score += r_t

            s_t1 = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
            memory.put((s_t,a_t[0],r_t,s_t1,done))
            s_t = s_t1

            if train_indicator and memory.size()>500:
                train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer)
                soft_update(mu, mu_target)
                soft_update(q,  q_target)
            
            log_timer+=1

            if done:
                break
        #t_end = timeit.default_timer()
        
        print("TOTAL REWARD @ " + str(n_epi) +"-th Episode  : Reward " + str(score))
        print("Total Step: " + str(n_step))
        print("")
        #print('{}steps, {} time spent'.format(i,t_end-t_start))
    env.end()
    # s,a,r,sp,d = memory.sample(1)
    # print('s: ',s)
    # print('a: ',a)
    # print('r: ',r)
    # print('sp: ', sp)
    # print('d: ',d)

def train(mu, mu_target, q, q_target, memory, q_optimizer, mu_optimizer):
    global log_timer
    s,a,r,s_prime,done_mask  = memory.sample(batch_size)
    # target = r if done
    target = r + np.logical_not(done_mask) *gamma*q_target(s_prime, mu_target(s_prime))
    q_loss = F.smooth_l1_loss(q(s,a), target.detach())
    q_optimizer.zero_grad()     #backward()가 가중치를 덮어씌우지 않고, 누적하기 때문에 초기화 한다.
    q_loss.backward()
    q_optimizer.step()
    
    mu_loss = -q(s,mu(s)).mean() # That's all for the policy loss.
    mu_optimizer.zero_grad()
    mu_loss.backward()
    mu_optimizer.step()

def soft_update(net, net_target):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)

if __name__ == '__main__':
    main()