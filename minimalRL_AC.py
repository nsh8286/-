#original code : https://github.com/seungeunrho/minimalRL/blob/master/actor_critic.py

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

#Hyperparameters
learning_rate = 0.0002
gamma         = 0.98
n_rollout     = 10

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.data = []
        
        self.fc1 = nn.Linear(4,256)
        self.fc_pi = nn.Linear(256,2)
        self.fc_v = nn.Linear(256,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
    def pi(self, x, softmax_dim = 0):                   #입력 차원에 따라 dim을 바꾼다
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob
    
    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v
    
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
        for transition in self.data:
            s,a,r,s_prime,done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r/100.0])                     #왜 100으로 나눠줬나? normalization?
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_lst.append([done_mask])                #done_mask의 존재 이유? 곱해주려고.
        
        s_batch, a_batch, r_batch, s_prime_batch, done_batch = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                                               torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_prime_lst, dtype=torch.float), \
                                                               torch.tensor(done_lst, dtype=torch.float)
        self.data = []
        return s_batch, a_batch, r_batch, s_prime_batch, done_batch
  
    def train_net(self):
        s, a, r, s_prime, done = self.make_batch()      #torch.tensor형태로 받아온다. 동시에 데이터 비움
        td_target = r + gamma * self.v(s_prime) * done  #done=true였다면 TD target = r. 일괄 계산
        delta = td_target - self.v(s)                   #delta 일괄 계산.
        
        pi = self.pi(s, softmax_dim=1)                  #policy 계산 (마지막에 행마다 softmax해주기)
        pi_a = pi.gather(1,a)                           #행 기준으로, index a 부분만 모으기 = pi(s,a)
        #detach() : backpropagation할 때 미분에 포함되는 것 방지. .item()효과
        #smooth_l1_loss : loss function중의 하나로 ㅣ1 loss와 ㅣ2 loss의 중간적인 느낌
        #아래식 앞 항은 policy 학습용, 뒷 항은 v(s) 학습용
        loss = -torch.log(pi_a) * delta.detach() + F.smooth_l1_loss(self.v(s), td_target.detach())

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()         
      
def main():  
    env = gym.make('CartPole-v1')
    model = ActorCritic()    
    print_interval = 20
    score = 0.0

    for n_epi in range(10000):
        done = False
        s = env.reset()
        while not done:
            for t in range(n_rollout):
                prob = model.pi(torch.from_numpy(s).float())    #policy 계산
                m = Categorical(prob)                           #sampling
                a = m.sample().item()                           #...
                s_prime, r, done, info = env.step(a)            #...
                model.put_data((s,a,r,s_prime,done))            #데이터 저장
                
                s = s_prime
                score += r
                
                if done:                                         
                    break                     
            
            model.train_net()                                   #끝, 또는 rollout 다 되면 학습
            
        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            score = 0.0
    env.close()

if __name__ == '__main__':
    main()