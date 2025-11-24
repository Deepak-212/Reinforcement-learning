import gymnasium as gym
import time
import numpy as np


# Create environment with human rendering
env = gym.make("CartPole-v1", render_mode="human")
episodeNumber = 5
timeSteps = 5
def Random_games():
    for episode in range(episodeNumber):  # play 10 episodes
        observation, info = env.reset()
        
        for t in range(timeSteps):
            action = env.action_space.sample()  # take random action
            observation, reward, terminated, truncated, info = env.step(action)
            time.sleep(0.1)
            print(t, observation, reward, terminated, truncated, info, action)
            if terminated or truncated:
                time.sleep(3)
                break

Random_games()

