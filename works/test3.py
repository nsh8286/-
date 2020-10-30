import torch
from minimalDDPG import MuNet # for one element tensor
from criticnetwork import QNet
import numpy as np
from ou import OrnsteinUhlenbeckNoise as OU
# --model에 어떤 형태의 data가 들어가야 하는지 테스트합니다.--
mu = MuNet()
q = QNet(state_dim=3,action_dim=1)
ou_noise = OU(np.zeros(1))
s = torch.rand(1,3)
a1 = mu(s)
print("a1: ",a1)
a2 = a1.item() + ou_noise()[0]
print("a2: {},{}".format(type(a2),a2))
print("ou_noise: {},{}".format(type(ou_noise),ou_noise))
a3 = np.float(a2)
print("a3: {},{}".format(type(a3),a3))
# # q에 a2를 입력해봅니다.
# # 'numpy.float64' object has no attribute 'dim' 오류가 발생합니다
# q_output = q(s,a2)
# print("q_output: ",q_output)

# 이 테스트는 불필요했습니다. minimal DDPG에서 sampling할 때 torch.tensor로 바꿔주고 있었습니다. 수고.
# 따라서 모델 데이터는 tensor형태여야 합니다.
# tensor는 꼭 2차원 형태여야 할까요?
a4 = torch.tensor([a2],dtype=torch.float)
print("a4: ",a4)

# # q에 a4를 넣어봅시다.1차원 tensor입니다.
# q_output = q(s,a4)
# print("q_output: ",q_output)
# #넣는 것 까지는 됩니다. 하지만 cat할 때 모양이 달라 에러가 발생합니다.

# 2차원 tensor를 넣어봅시다.
a5 = torch.tensor([[a2]],dtype=torch.float)
q_output = q(s,a5)
print("q_output: ",q_output)
# it works!