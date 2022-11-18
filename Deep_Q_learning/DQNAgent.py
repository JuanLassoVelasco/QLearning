
import numpy as np
import gym
import torch
import torch.nn as nn
from torch.nn.modules.activation import Sigmoid
from torch.nn import Sequential
from matplotlib import pyplot as plt

class DQnn(nn.Module):
    def __init__(self, inputNodeNum, outputNodeNum, num_hid_layers, num_neurons):
        super().__init__()
        self.numInput = inputNodeNum
        self.numOutput = outputNodeNum
        self.numHidLayers = num_hid_layers
        self.numNeurons = num_neurons

        self.layers = Sequential(
            nn.Linear(inputNodeNum, num_neurons)
        )

        for i in range(0, num_hid_layers):
            self.layers.append(nn.Sigmoid())
            self.layers.append(nn.Linear(num_neurons, num_neurons))

        self.layers.append(nn.Sigmoid())
        self.layers.append(nn.Linear(num_neurons, outputNodeNum))

    def forward(self, x):
        return self.layers(x)

class DQNAgent:
    def __init__(self, observSize, actionSize, lr=0.001, dr=0.99, batchSize=30, mem_buffer_max=2000, explor_prob_decay=0.0005):
        self.action_spc_size = actionSize
        self.obs_spc_size = observSize
        self.learn_rate = lr
        self.discount_rate = dr
        self.explore_prob = 1.0
        self.exp_prob_decay = explor_prob_decay
        self.batch_size = batchSize
        self.memory_buffer = list()
        self.max_memory_buffer = mem_buffer_max

        self.model = DQnn(observSize, actionSize, 4, 50)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learn_rate)

    def determineAction(self, currentState):
        if np.random.uniform(0,1) < self.explore_prob:
            return np.random.choice(range(self.action_spc_size))

        Qvalues = self.model(torch.from_numpy(currentState).float())

        return np.argmax(Qvalues.detach().numpy())

    def decreaseExploreProb(self):
        self.explore_prob = self.explore_prob * np.exp(-self.exp_prob_decay)

    def storeExperience(self, currentState, action, reward, nextState, done):
        self.memory_buffer.append({
            "current_state": currentState,
            "action": action,
            "reward": reward,
            "next_state":nextState,
            "done": done
        })

        if len(self.memory_buffer) > self.max_memory_buffer:
            self.memory_buffer.pop(0)

    def trainDQnn(self):
        np.random.shuffle(self.memory_buffer)
        batchSample = self.memory_buffer[0:self.batch_size]

        for exp in batchSample:
            self.optimizer.zero_grad()
            stateQ = self.model(torch.from_numpy(exp["current_state"]).float())
            predQ = stateQ

            targetQ = exp["reward"]

            if not exp["done"]:
                targetQ = targetQ + self.discount_rate*torch.max(self.model(torch.from_numpy(exp["next_state"]).float()))

            stateQ[exp["action"]] = targetQ

            loss = self.criterion(predQ, stateQ)
            
            loss.backward()
            self.optimizer.step()
