import numpy as np
from enum import Enum

from soupsieve import select
from Eater import Eater

class MapItem(Enum):
    FOODBIT = 1
    FOODCLUSTER = 2
    ENEMY = 3

class Map:
    def __init__(self, rows, cols, spacePenalty=-0.5, enemyPenalty=-50, foodReward=5, foodClusterReward=40, startPoseX = 0, startPoseY = 0):
        self.agent = 0
        self.agentPose = [startPoseX, startPoseY]
        self.numRows = rows
        self.numCols = cols
        self.singleFoodCount = 0
        self.foodClusterCount = 0
        self.enemyCount = 0
        self.spacePenalty = spacePenalty
        self.enemyPenalty = enemyPenalty
        self.foodReward = foodReward
        self.foodClusterReward = foodClusterReward
        self.endConditionMet = False

        self.outOfBoundPenalty = -40

        self.map = np.zeros((self.numRows, self.numCols))

        # initialized blank space rewards except start
        for x in range(0, cols):
            for y in range(0, rows):
                if [x, y] == [startPoseX, startPoseY]:
                    continue

                self.map[x, y] = spacePenalty

    def placeEater(self, eater = Eater()):
        self.agentPose = [0, 0]
        self.agent = eater
        self.agent.setPose = self.agentPose

    def placeElement(self, mapItem, xpose, ypose):
        if mapItem == MapItem.FOODBIT:
            self.map[xpose, ypose] = self.foodReward
            self.singleFoodCount += 1
        elif mapItem == MapItem.FOODCLUSTER:
            self.map[xpose, ypose] = self.foodClusterReward
            self.foodClusterCount += 1
        else:
            self.map[xpose, ypose] = self.enemyPenalty
            self.enemyCount += 1

    def moveAgent(self, x, y):
        # check boundaries
        if x < 0 or x >= self.numRows:
            print("Agent attempted to move out of bounds")
            if x < 0:
                self.agentPose[0] = 0
            else:
                self.agentPose[0] = 9
            self.endConditionMet = True
            self.agent.updateReward(self.outOfBoundPenalty)
            self.agent.decreaseLifeSpan()
            return self.outOfBoundPenalty
        if y < 0 or y >= self.numCols:
            print("Agent attempted to move out of bounds")
            if y < 0:
                self.agentPose[1] = 0
            else:
                self.agentPose[1] = 9
            self.endConditionMet = True
            self.agent.updateReward(self.outOfBoundPenalty)
            self.agent.decreaseLifeSpan()
            return self.outOfBoundPenalty

        # move agent
        self.agentPose = [x, y]

        # check space for reward
        spaceReward = self.map[x, y]

        if spaceReward == self.enemyPenalty or spaceReward == self.foodClusterReward:
            self.endConditionMet = True
        elif spaceReward == self.foodReward:
            self.map[x, y] = self.spacePenalty

        # update agent
        self.agent.updateReward(spaceReward)
        self.agent.decreaseLifeSpan()

        if self.agent.isDead():
            self.endConditionMet = True

        return spaceReward

    def checkEndConditionMet(self):
        return self.endConditionMet

    def getAgentPose(self):
        return self.agentPose

    def render(self):

        for y in range(self.numRows - 1, 0 - 1, -1):
            print("|", end='')
            for x in range(0, self.numCols):
                if self.map[x, y] == self.enemyPenalty:
                    print("X|", end='')
                elif self.agentPose == [x, y]:
                    print("@|", end='')
                elif self.map[x, y] == self.foodReward:
                    print(".|", end='')
                elif self.map[x, y] == self.foodClusterReward:
                    print(":|", end='')
                else:
                    print(" |", end='')

            print("\n", end='')

