import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math
class RLEnv(gym.Env):
    def __init__(self,df):
        super(RLEnv, self).__init__()
        
        # Define hyper-parameters for action_history buffer
        self.action_history_length = 5  # x: Number of previous steps to remember
        self.action_dim = 1 
        self.action_history = [0.0] * self.action_history_length  # List of previous actions
        self.state=0
        self.current_step = 0
        self.max_steps = 1000  # Maximum steps per episode
        self.high_stress_threshold = 7.0
        self.low_stress_threshold = 3.0
        # Hidden state variables
        self.E_t = 0.0  # Physical effort
        self.S_t = 0.0  # Psychological stress
        # Define the Observation Space as flat array
        # [hr, hrv, vel, jitter, action_history[2], action_history[1], action_history[0]]
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(7,), 
            dtype=np.float32
        )
        
        # Action space: discrete actions (0 and 1)
        self.action_space = spaces.Discrete(2)
    def _get_obs(self):
        if self.df!=None:
            obs={
                "HR_t": self.df["HR_t"].values[self.current_step],
                "HRV_t": self.df["HRV_t"].values[self.current_step],
                "omega_head": self.df[["omega_head_x", "omega_head_y", "omega_head_z"]].values[self.current_step],
                "alpha_head": self.df[["alpha_head_x", "alpha_head_y", "alpha_head_z"]].values[self.current_step],
                "a_t_x": np.zeros((self.action_history_length, self.action_dim), dtype=np.float32)  # Placeholder for action action_history
            }
            return obs
        # This method would be called to get the current observation
        # For now, we return a dummy observation matching the space
        obs = {
            "HR_t": np.random.uniform(30.0, 220.0, size=(1,)).astype(np.float32),
            "HRV_t": np.random.uniform(0.0, 200.0, size=(1,)).astype(np.float32),
            "omega_head": np.random.uniform(-35.0, 35.0, size=(3,)).astype(np.float32),
            "alpha_head": np.random.uniform(-100.0, 100.0, size=(3,)).astype(np.float32),
            "a_t_x": np.random.uniform(-1.0, 1.0, size=(self.action_history_length, self.action_dim)).astype(np.float32)
        }
        return obs
    def reset(self, seed=None):
        super().reset(seed=seed)
        info = {}
        self.current_step = 0
        self.E_t = 0.0  # Reset effort
        self.S_t = 0.0  # Reset stress
        self.action_history = [0.0] * self.action_history_length  # Reset action history
        
        # Calculate initial observation matching step function format
        hr = 70 + (30 * self.E_t) + (30 * self.S_t) + np.random.normal(0, 1.5)
        hrv = 50 - (35 * self.S_t) + np.random.normal(0, 1.0)
        vel = max(0, 2.0 * self.E_t * (1 - 4 * (self.S_t - 0.5)**2) + np.random.normal(0, 0.1))
        jitter = 0.5 + math.exp(10 * (self.S_t - 0.8)) if self.S_t > 0.8 else 0.5 + np.random.normal(0, 0.1)
        
        obs = np.array([hr, hrv, vel, jitter, self.action_history[2], self.action_history[1], self.action_history[0]], dtype=np.float32)
        return obs, info
    
    def step(self, action):
        self.current_step += 1
        
        # 1. Update Hidden Variables based on Action
        # Action 2 (Escalate) increases BOTH psychological stress and physical effort (more nodes to grab)
        # Action 0 (De-escalate) lowers BOTH.
        recent_uses = sum(1 for a in self.action_history if a == action)
        decay = math.exp(-0.5 * recent_uses)
        print(action)
        if action == 0: 
            delta_S = -0.05
            self.E_t = max(0.0, self.E_t - 0.1) # Less physical effort needed
        elif action == 1: 
            delta_S = 0.0
        elif action == 2: 
            delta_S = 0.1
            self.E_t = min(1.0, self.E_t + 0.1) # More physical effort needed
            
        homeostasis = 0.05 * (self.S_t - 0.2)
        self.S_t = np.clip(self.S_t + (delta_S * decay) - homeostasis + np.random.normal(0, 0.02), 0.0, 1.0)
        
        # 2. The Biologically Accurate Sensor Mapping
        # HR goes up from BOTH physical effort AND psychological stress
        hr = 70 + (30 * self.E_t) + (30 * self.S_t) + np.random.normal(0, 1.5)
        
        # HRV drops ONLY from psychological stress. Physical effort doesn't ruin HRV.
        hrv = 50 - (35 * self.S_t) + np.random.normal(0, 1.0) 
        
        # Velocity peaks when engaged (E_t is high, but S_t is moderate)
        vel = max(0, 2.0 * self.E_t * (1 - 4 * (self.S_t - 0.5)**2) + np.random.normal(0, 0.1))
        
        # Jitter triggers ONLY from extreme psychological stress
        jitter = 0.5 + math.exp(10 * (self.S_t - 0.8)) if self.S_t > 0.8 else 0.5 + np.random.normal(0, 0.1)

        # 3. THE CORRECTED REWARD FUNCTION
        # Maximize task engagement (Velocity)
        # Maximize calm/flow state (HRV)
        # Minimize panic (Jitter)
        # Notice: HR is NOT in this equation. The AI is free to push HR up to 115 if it's purely from Effort!
        reward = (1.5 * vel) + (0.5 * (hrv / 50.0)) - (2.0 * jitter)
        
        # 4. The Symbolic Veto (Terminal State)
        terminated = False
        truncated = False
        
        # The AI only hits the veto wall if it pushes HR to dangerous absolute limits (120) 
        # OR if it crashes the HRV, proving it induced panic instead of exercise.
        if hr > 120.0 or hrv < 15.0 or jitter > 2.5:
            reward -= 100.0 
            terminated = True
            
        if self.current_step >= self.max_steps:
            truncated = True 

        # 5. Update action_history & State Vector
        self.action_history.append(action)
        self.action_history.pop(0)
        
        obs = np.array([hr, hrv, vel, jitter, self.action_history[2], self.action_history[1], self.action_history[0]], dtype=np.float32)
        
        return obs, reward, terminated, truncated, {}
    
    # def step(self, action):
    #     """
    #     Applies an action, steps the environment forward by one tick, and returns the new state.
    #     """
    #     self.current_step += 1
    #     self.action_history.pop(0)
    #     self.action_history.append(action)
    #     # 1. Calculate the new state based on the action
    #     # Example: The action changes the state directly

    #     # 2. Calculate the reward
    #     # Example: Reward the agent for getting closer to a target value of 5.0
    #     current_stress=self.df[self.current_step]['stress']
    #     direction = self.df[self.current_step]['gradient']
        
    #     if(current_stress > self.high_stress_threshold and action>0):
    #         reward = -1.0  # Penalize high stress
    #     elif(current_stress < self.low_stress_threshold and action<0):
    #         reward = -1.0  # Penalize low stress
    #     elif(current_stress > self.high_stress_threshold and action<0):
    #         reward = 1.0  # Reward reducing high stress
    #     elif(current_stress < self.low_stress_threshold and action>0):
    #         reward = 1.0  # Reward increasing low stress
    #     else:
    #         reward = 0.0  # No reward for neutral actions
        
    #     # 3. Check if the episode is over
    #     terminated = False # True if the agent reaches the goal
    #     truncated = False # True if a time limit is reached

    #     info = {}

    #     return self.state, float(reward), terminated, truncated, info