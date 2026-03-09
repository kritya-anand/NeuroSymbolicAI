import gym
from Environment import RLEnv
import numpy as np
from gym.envs import registration



# Dummy DataFrame for testing (replace with real data in practice)
import pandas as pd
data = {
    'HR_t': np.random.uniform(60, 100, 10),
    'HRV_t': np.random.uniform(20, 80, 10),
    'omega_head_x': np.random.uniform(-10, 10, 10),
    'omega_head_y': np.random.uniform(-10, 10, 10),
    'omega_head_z': np.random.uniform(-10, 10, 10),
    'alpha_head_x': np.random.uniform(-5, 5, 10),
    'alpha_head_y': np.random.uniform(-5, 5, 10),
    'alpha_head_z': np.random.uniform(-5, 5, 10),
    'stress': np.random.uniform(1, 10, 10),
    'gradient': np.random.uniform(-1, 1, 10)
}
df = pd.DataFrame(data)

env = RLEnv(df)

# Register the environment with gym (optional, for gym.make usage)
gym.envs.registration.register(
    id='RLEnv-v0',
    entry_point='Environment:RLEnv',
    kwargs={'df': df}
)

# Test reset and step
obs, info = env.reset()
print('Reset observation:', obs)

for i in range(3):
    action = env.action_space.sample()
    result = env.step(action)
    print(f'Step {i+1} result:', result)
