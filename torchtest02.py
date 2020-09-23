import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

z = torch.FloatTensor([99,100,101])
hypothesis = F.softmax(z, dim=0)

print(hypothesis)