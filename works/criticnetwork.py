import torch
import torch.nn as nn
import torch.nn.functional as F

class QNet(nn.Module):          #critic
    def __init__(self,state_dim,action_dim):
        super(QNet, self).__init__()
        self.fc_s1 = nn.Linear(state_dim, 256)
        self.fc_s2 = nn.Linear(256, 256)
        self.fc_a1 = nn.Linear(action_dim,128)
        self.fc_a2 = nn.Linear(128,128)
        self.fc_q = nn.Linear(384, 64)
        self.fc_3 = nn.Linear(64,1)

    def forward(self, x, a):
        h11 = F.relu(self.fc_s1(x))
        h12 = F.relu(self.fc_s2(h11))
        h21 = F.relu(self.fc_a1(a))
        h22 = F.relu(self.fc_a2(h21))
        cat = torch.cat([h12,h22], dim=1) #64개 parameter들로 나눈 state + action 합치기
        q = F.relu(self.fc_q(cat))
        q = self.fc_3(q)
        # print("------CriticNet start-------")
        # print("cat: ",cat)
        # print("q: ", q)
        # print("------CriticNet close-------")
        return q