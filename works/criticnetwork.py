import torch
import torch.nn as nn
import torch.nn.functional as F

class QNet(nn.Module):          #critic
    def __init__(self,state_dim,action_dim):
        super(QNet, self).__init__()
        self.fc_s1 = nn.Linear(state_dim, 256)
        self.fc_s2 = nn.Linear(256, 512)
        self.fc_a = nn.Linear(action_dim,512)
        self.fc_m = nn.Linear(512,512)
        self.fc_q = nn.Linear(512, 1)

    def forward(self, x, a):
        h11 = F.relu(self.fc_s1(x))
        h12 = self.fc_s2(h11)
        h2 = self.fc_a(a)
        merge = h12 +h2 #64개 parameter들로 나눈 state + action 합치기
        q1 = F.relu(self.fc_m(merge))
        q2 = self.fc_q(q1)
        # print("------CriticNet start-------")
        # print("cat: ",cat)
        # print("q: ", q)
        # print("------CriticNet close-------")
        return q2