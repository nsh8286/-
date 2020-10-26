class MuNet(nn.Module):         #actor
    def __init__(self,state_dim,action_dim):
        super(MuNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)    #입력은 state_dim
        self.fc2 = nn.Linear(128, 64)
        self.fc_str = nn.Linear(64, 1)  #steering output
        self.fc_acc = nn.Linear(64, 1)  #accelerator output

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x))*2 # Multipled by 2 because the action space of the Pendulum-v0 is [-2,2]
        return mu