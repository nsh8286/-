import torch
import torch.nn as nn
import torch.nn.functional as F
class MuNet(nn.Module):         #actor
    def __init__(self,state_dim):
        super(MuNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)    #입력은 state_dim
        self.fc2 = nn.Linear(256, 512)
        self.fc_steer = nn.Linear(512, 1)  #steering output
        self.fc_acc = nn.Linear(512, 1)  #accelerator output

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        steer = torch.tanh(self.fc_steer(x))#[-1,1]
        acc = torch.sigmoid(self.fc_acc(x))#[0,1]
        cat = torch.cat([steer,acc],dim=1) #bind actions together

        # print("------ActorNet start-------")
        # print("steer: ",steer)
        # print("acc: ",acc)
        # print("cat: ",cat)
        # print("------ActorNet close-------")
        return cat