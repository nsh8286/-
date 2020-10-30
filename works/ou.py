# refer to https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
import numpy as np

class OrnsteinUhlenbeckNoise(object):           
    def __init__(self, mu, theta=0.1, dt=0.01, sigma=0.1, x0=None):
        self.theta, self.dt, self.sigma = theta, dt, sigma
        self.mu = mu
        self.x0 = x0
        self.reset()    #mu와 같은 모양의 0으로 이루어진 배열

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x
    
    def __repr__(self):
        return '{}'.format(self.x_prev)

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)
      
