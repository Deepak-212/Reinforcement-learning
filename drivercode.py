import gymnasium as gym
import numpy as np
import time
from Q_learning import Q_learning
import os

env=gym.make('CartPole-v1')
(state,_)=env.reset()

upperBounds=env.observation_space.high
lowerBounds=env.observation_space.low
cartVelocityMin=-3
cartVelocityMax=3
poleAngleVelocityMin=-10
poleAngleVelocityMax=10
upperBounds[1]=cartVelocityMax
upperBounds[3]=poleAngleVelocityMax
lowerBounds[1]=cartVelocityMin
lowerBounds[3]=poleAngleVelocityMin

numberOfBinsPosition=30
numberOfBinsVelocity=30
numberOfBinsAngle=30
numberOfBinsAngleVelocity=30
numberOfBins=[numberOfBinsPosition,numberOfBinsVelocity,numberOfBinsAngle,numberOfBinsAngleVelocity]

alpha=0.1
gamma=1
epsilon=0.2
numberEpisodes=8000
# -----------------------------
# File for saving Q-matrix
# -----------------------------
qmatrix_file = "trained_Qmatrix.npy"

# -----------------------------
# Train if no saved Q-matrix exists, otherwise load
# -----------------------------
if not os.path.exists(qmatrix_file):
    print("No saved Q-matrix found. Training now...")
    Q1 = Q_learning(env, alpha, gamma, epsilon, numberEpisodes, numberOfBins, lowerBounds, upperBounds)
    Q1.simulateEpisodes()
    np.save(qmatrix_file, Q1.Qmatrix)
    print("Training complete. Q-matrix saved.")
else:
    print("Found saved Q-matrix. Loading...")
    Q1 = Q_learning(env, alpha, gamma, epsilon, numberEpisodes, numberOfBins, lowerBounds, upperBounds)
    Q1.Qmatrix = np.load(qmatrix_file)
    obtainedRewardsOptimal, env1 = Q1.simulateLearnedStrategy()

    print("Rewards from learned strategy:", obtainedRewardsOptimal)
