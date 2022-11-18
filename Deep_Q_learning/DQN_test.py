import DQNAgent
from DQNAgent import DQNAgent
import numpy as np
import gym
import torch
from gym.wrappers import record_video


def make_video():
    env_to_wrap = gym.make('CartPole-v1', render_mode="rgb_array")
    env = record_video.RecordVideo(env_to_wrap, 'video')
    rewards = 0
    steps = 0
    done = False
    state = env.reset()
    state, _ = state
    with torch.no_grad():
        while not done:
            action = agent.determineAction(state)
            state, reward, terminated, truncated, _ = env.step(action) 
            done = terminated or truncated      
            steps += 1
            rewards += reward
    print(rewards)
    env.close()
    env_to_wrap.close()

# We create our gym environment 
env = gym.make("CartPole-v1")
# We get the shape of a state and the actions space size
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
print("observation space:" + str(state_size))
# Number of episodes to run
n_episodes = 1000
# Max iterations per epiode
max_iteration_ep = 500
# We define our agent
learnRate = 0.001
discountRate = 0.99
batchSize = 30
agent = DQNAgent(state_size, action_size, lr=learnRate, dr=discountRate, batchSize=batchSize)
total_steps = 0

torch.autograd.set_detect_anomaly(True)

# We iterate over episodes
for e in range(n_episodes):
    # We initialize the first state and reshape it to fit 
    #  with the input layer of the DNN
    current_state = env.reset()
    current_state, _ = current_state
    for step in range(max_iteration_ep):
        total_steps = total_steps + 1
        # the agent computes the action to perform
        action = agent.determineAction(current_state)
        # the envrionment runs the action and returns
        # the next state, a reward and whether the agent is done
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # We sotre each experience in the memory buffer
        agent.storeExperience(current_state, action, reward, next_state, done)
        
        # if the episode is ended, we leave the loop after
        # updating the exploration probability
        if done:
            agent.decreaseExploreProb()
            break
        current_state = next_state
    # if the have at least batch_size experiences in the memory buffer
    # than we tain our model
    if total_steps >= batchSize:
        agent.trainDQnn()

make_video()
