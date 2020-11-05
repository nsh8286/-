import torch
import torch.nn as nn
import torch.nn.functional as F

class QNet(nn.Module):          #critic
    def __init__(self,state_dim,action_dim):
        super(QNet, self).__init__()
        self.fc_s = nn.Linear(state_dim, 64)
        self.fc_a = nn.Linear(action_dim,64)
        self.fc_q = nn.Linear(128, 32)
        self.fc_3 = nn.Linear(32,1)

    def forward(self, x, a):
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1,h2], dim=1) #64개 parameter들로 나눈 state + action 합치기
        q = F.relu(self.fc_q(cat))
        q = self.fc_3(q)
        # print("------CriticNet start-------")
        # print("cat: ",cat)
        # print("q: ", q)
        # print("------CriticNet close-------")
        return q