import gym
from gym.envs.registration import register
import msvcrt

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

arrow_keys = {
    72: UP,
    80 : DOWN,
    77: RIGHT,
    75 : LEFT
}

register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False}
)

class _Getch:
    def __call__(self):
        return (ord(msvcrt.getch()),ord(msvcrt.getch()))
        #두번 받아오고 튜플로 내보낸다?

inkey = _Getch()

env = gym.make("FrozenLake-v3")
env.render()
observation = env.reset()

while True:
    
    key = inkey()
    if key[0] != 224 and key[1] not in arrow_keys.keys():
        print("Game aborted!")
        break
    
    action = arrow_keys[key[1]]
    state, reward, done, info = env.step(action)
    env.render()
    print("State: {0}, Action: {1}, Reward: {2}, Info: {3}".format(state, action, reward, info))

    if done:
        print("Finished with reward", reward)
        break


