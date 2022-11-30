from time import sleep
import numpy as np
import gym
from gym.wrappers import record_video
from DQN_Agent import DQN_Agent

env = gym.make('CartPole-v1')
input_size = env.observation_space.shape[0]
output_size = env.action_space.n
exp_storage_size = 256

learnRate = 1e-3
hid_layer_n = 64
score_disp_interval = 100

agent = DQN_Agent(seed = 1423, layer_sizes = [input_size, hid_layer_n, output_size], lr = learnRate, sync_freq = 5, exp_replay_size = exp_storage_size)

# initiliaze experiance replay      
index = 0
for i in range(exp_storage_size):
    obs = env.reset()
    obs, _ = obs
    done = False
    while(done != True):
        action = agent.get_action(obs, env.action_space.n, epsilon=1)
        obs_next, reward, terminate, truncate, _ = env.step(action.item())
        done = terminate or truncate
        agent.collect_experience([obs, action.item(), reward, obs_next])
        obs = obs_next
        index += 1
        if( index > exp_storage_size ):
            break
            
# Main training loop
losses_list, reward_list, episode_len_list, epsilon_list  = [], [], [], []
index = 128
episodes = 10000
epsilon = 1

for i in range(episodes):
    obs, done, losses, ep_len, rew = env.reset(), False, 0, 0, 0
    obs, _ = obs
    while(done != True):
        ep_len += 1 
        action = agent.get_action(obs, env.action_space.n, epsilon)
        obs_next, reward, terminate, truncate, _ = env.step(action.item())
        done = terminate or truncate
        agent.collect_experience([obs, action.item(), reward, obs_next])
       
        obs = obs_next
        rew  += reward
        index += 1
        
        if(index > 128):
            index = 0
            for j in range(4):
                loss = agent.train(batch_size=16)
                losses += loss      
    if epsilon > 0.05 :
        epsilon -= (1 / 5000)

    if i % score_disp_interval == 0:
        print('Episode {}\tAverage Score: {:.2f}'.format(i, rew))
    
    losses_list.append(losses/ep_len), reward_list.append(rew), episode_len_list.append(ep_len), epsilon_list.append(epsilon)

env_to_wrap = gym.make('CartPole-v1', render_mode="rgb_array")
env = record_video.RecordVideo(env_to_wrap, 'video')
for i in range(2):
    obs, done, rew = env.reset(), False, 0
    obs, _ = obs
    while (done != True) :
        action =  agent.get_action(obs, env.action_space.n, epsilon = 0)
        obs, reward, terminate, truncate, info = env.step(action.item())
        done = terminate or truncate
        rew += reward
        sleep(0.01)
    print("episode : {}, reward : {}".format(i,rew)) 

    print(rew)
    env.close()
    env_to_wrap.close()
