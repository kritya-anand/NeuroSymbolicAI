import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from Environment import RLEnv
import os

class TrainingCallback(BaseCallback):
    """Custom callback for tracking training progress"""
    def __init__(self, verbose=0):
        super(TrainingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0
        
    def _on_step(self) -> bool:
        # Get info from the environment
        if "episode" in self.model.env.get_attr("_episode_rewards"):
            pass
        return True

class PPOTrainer:
    def __init__(self, env, num_timesteps=100000, learning_rate=3e-4, n_steps=2048):
        self.env = env
        self.num_timesteps = num_timesteps
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        
        print(f"Initializing PPO trainer with stable-baselines3")
        print(f"Environment action space: {self.env.action_space}")
        print(f"Environment observation space shape: 7 (flat array)")
        
        # Initialize PPO model
        self.model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            verbose=1,
            device="auto"
        )
        
        self.episode_rewards = []
        self.episode_lengths = []
        
    def train(self):
        """Train the agent using PPO from stable-baselines3"""
        print(f"\nStarting PPO training for {self.num_timesteps} timesteps...")
        print("-" * 100)
        
        # Train the model
        self.model.learn(total_timesteps=self.num_timesteps)
        
        print("-" * 100)
        print("Training Complete!")
        
    def evaluate(self, num_episodes=10):
        """Evaluate the trained model"""
        print(f"\nEvaluating model for {num_episodes} episodes...")
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            obs, info = self.env.reset()
            episode_reward = 0.0
            episode_length = 0
            done = False
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                episode_length += 1
                done = terminated or truncated
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            print(f"Episode {episode + 1:3d} | Reward: {episode_reward:8.2f} | Length: {episode_length:5d}")
        
        return episode_rewards, episode_lengths
    
    def collect_episode_data(self, num_episodes=10, output_dir="episode_logs"):
        """Collect and log observations, actions, and rewards for each episode"""
        print(f"\nCollecting episode data for {num_episodes} episodes...")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        episode_summaries = []
        
        for episode in range(num_episodes):
            obs, info = self.env.reset()
            done = False
            step = 0
            
            # Data for this episode
            episode_data = {
                'step': [],
                'observation': [],
                'action': [],
                'reward': [],
                'hr': [],
                'hrv': [],
                'velocity': [],
                'jitter': [],
                'action_hist_1': [],
                'action_hist_2': [],
                'action_hist_3': []
            }
            
            episode_reward = 0.0
            
            while not done:
                # Get action from model
                action, _ = self.model.predict(obs, deterministic=True)
                
                # Store observation components
                hr, hrv, vel, jitter = obs[0], obs[1], obs[2], obs[3]
                act_hist_1, act_hist_2, act_hist_3 = obs[4], obs[5], obs[6]
                
                # Take step
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Log data
                episode_data['step'].append(step)
                episode_data['observation'].append(str(obs))
                episode_data['action'].append(int(action))
                episode_data['reward'].append(float(reward))
                episode_data['hr'].append(float(hr))
                episode_data['hrv'].append(float(hrv))
                episode_data['velocity'].append(float(vel))
                episode_data['jitter'].append(float(jitter))
                episode_data['action_hist_1'].append(float(act_hist_1))
                episode_data['action_hist_2'].append(float(act_hist_2))
                episode_data['action_hist_3'].append(float(act_hist_3))
                
                episode_reward += reward
                obs = next_obs
                step += 1
            
            # Save episode data to CSV
            df_episode = pd.DataFrame(episode_data)
            episode_file = os.path.join(output_dir, f"episode_{episode + 1:03d}.csv")
            df_episode.to_csv(episode_file, index=False)
            
            # Track summary
            episode_summaries.append({
                'episode': episode + 1,
                'total_steps': step,
                'total_reward': episode_reward,
                'avg_reward_per_step': episode_reward / step if step > 0 else 0
            })
            
            print(f"Episode {episode + 1:3d} | Steps: {step:4d} | Total Reward: {episode_reward:8.2f} | "
                  f"Avg Reward/Step: {episode_reward/step if step > 0 else 0:8.2f} | "
                  f"Saved to {episode_file}")
        
        # Save summary
        df_summary = pd.DataFrame(episode_summaries)
        summary_file = os.path.join(output_dir, "episode_summary.csv")
        df_summary.to_csv(summary_file, index=False)
        print(f"\nEpisode summary saved to {summary_file}")
        
        return episode_summaries
    
    def print_summary(self, eval_rewards=None):
        """Print summary statistics"""
        print("\n=== PPO Training Summary ===")
        if eval_rewards:
            print(f"Evaluation Episodes: {len(eval_rewards)}")
            print(f"Average Evaluation Reward: {np.mean(eval_rewards):.2f}")
            print(f"Max Evaluation Reward: {np.max(eval_rewards):.2f}")
            print(f"Min Evaluation Reward: {np.min(eval_rewards):.2f}")
            print(f"Std Dev Evaluation Reward: {np.std(eval_rewards):.2f}")
    
    def save_model(self, path="ppo_model"):
        """Save the trained model"""
        self.model.save(path)
        print(f"\nModel saved to {path}")
    
    def load_model(self, path="ppo_model"):
        """Load a trained model"""
        self.model = PPO.load(path)
        print(f"Model loaded from {path}")


if __name__ == "__main__":
    # Create environment
    env = RLEnv(df=None)
    
    # Create and run PPO trainer using stable-baselines3
    trainer = PPOTrainer(env, num_timesteps=100000, learning_rate=3e-4, n_steps=2048)
    
    # Train the model
    trainer.train()
    
    # Save the model
    trainer.save_model("ppo_trained_model")
    
    # Collect episode data with step-by-step logging
    print("\n" + "="*100)
    print("Collecting detailed episode data...")
    print("="*100)
    episode_summaries = trainer.collect_episode_data(num_episodes=10, output_dir="episode_logs")
    
    # Print summary
    trainer.print_summary(eval_rewards=[s['total_reward'] for s in episode_summaries])
    
    # Create and display summary dataframe
    df_summary = pd.DataFrame(episode_summaries)
    print("\nEpisode Summary:")
    print(df_summary)
    
    # Save evaluation data
    df_summary.to_csv('ppo_evaluation_results.csv', index=False)
    print("\nAll episode data saved to 'episode_logs/' directory")
    print("Episode summary saved to 'ppo_evaluation_results.csv'")
