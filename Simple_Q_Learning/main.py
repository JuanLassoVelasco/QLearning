from sqlalchemy import false
from Map import Map
from Map import MapItem
from Eater import Eater
from QLearner import QLearner
import numpy as np
import keyboard
import os
import time

if __name__ == "__main__":

    numEpisodes = 10000
    gridSizeX = 10
    gridSizeY = 10

    actionSpaceSize = 4
    discountRate = 0.99
    learnRate = 0.7

    qlearner = QLearner(gridXSize=gridSizeX, gridYSize=gridSizeY, numActions=actionSpaceSize, learnRate=learnRate, discountRate=discountRate)

    for i in range(0, numEpisodes):
        agent = Eater()
        agentX = 0
        agentY = 0
        worldMap = Map(gridSizeY, gridSizeX)

        worldMap.placeElement(MapItem.ENEMY, 4, 4)
        worldMap.placeElement(MapItem.ENEMY, 8, 1)
        worldMap.placeElement(MapItem.ENEMY, 1, 8)
        worldMap.placeElement(MapItem.ENEMY, 5, 0)
        worldMap.placeElement(MapItem.ENEMY, 6, 8)
        
        worldMap.placeElement(MapItem.FOODBIT, 3, 7)
        worldMap.placeElement(MapItem.FOODBIT, 7, 3)
        worldMap.placeElement(MapItem.FOODBIT, 9, 7)
        worldMap.placeElement(MapItem.FOODBIT, 3, 3)
        worldMap.placeElement(MapItem.FOODBIT, 5, 7)
        worldMap.placeElement(MapItem.FOODBIT, 2, 5)
        worldMap.placeElement(MapItem.FOODBIT, 6, 1)
        worldMap.placeElement(MapItem.FOODBIT, 1, 2)
        worldMap.placeElement(MapItem.FOODBIT, 2, 0)
        worldMap.placeElement(MapItem.FOODBIT, 0, 3)
        worldMap.placeElement(MapItem.FOODBIT, 3, 1)
        worldMap.placeElement(MapItem.FOODBIT, 1, 4)
        worldMap.placeElement(MapItem.FOODBIT, 4, 2)
        worldMap.placeElement(MapItem.FOODBIT, 5, 5)
        worldMap.placeElement(MapItem.FOODBIT, 7, 5)

        worldMap.placeElement(MapItem.FOODCLUSTER, 8, 7)

        worldMap.placeEater(agent)
        # Actions:
        #   Left = 0
        #   Right = 1
        #   Up = 2
        #   Down = 3

        while worldMap.checkEndConditionMet() == False:
            # determine next action
            nextAction = qlearner.QLearnStep(agentXPose=agentX, agentYPose=agentY)

            curXpose = agentX
            curYpose = agentY

            # execute action
            if nextAction == 0:
                agentX -= 1
            elif nextAction == 1:
                agentX += 1
            elif nextAction == 2:
                agentY += 1
            elif nextAction == 3:
                agentY -= 1

            if agentX < 0:
                agentX = 0
            elif agentX >= gridSizeX:
                agentX = gridSizeX - 1

            if agentY < 0:
                agentY = 0
            elif agentY >= gridSizeY:
                agentY = gridSizeY - 1

            actionReward = worldMap.moveAgent(x=agentX, y=agentY)

            # update qtable
            qlearner.updateQTable(xpos=curXpose, ypos=curYpose, xfuture=agentX, yfuture=agentY, action=nextAction, reward=actionReward)

            
            
        totalReward = agent.getTotalReward()
        print("Total reward for episode {}:  {}".format(i, totalReward))
        if i == numEpisodes - 1:
            worldMap.render()
    
    # print(qlearner.Qtable)

    agent = Eater()
    agentX = 0
    agentY = 0
    worldMap = Map(gridSizeY, gridSizeX)

    worldMap.placeElement(MapItem.ENEMY, 4, 4)
    worldMap.placeElement(MapItem.ENEMY, 8, 1)
    worldMap.placeElement(MapItem.ENEMY, 1, 8)
    worldMap.placeElement(MapItem.ENEMY, 5, 0)
    worldMap.placeElement(MapItem.ENEMY, 6, 8)
    
    worldMap.placeElement(MapItem.FOODBIT, 3, 7)
    worldMap.placeElement(MapItem.FOODBIT, 7, 3)
    worldMap.placeElement(MapItem.FOODBIT, 9, 7)
    worldMap.placeElement(MapItem.FOODBIT, 3, 3)
    worldMap.placeElement(MapItem.FOODBIT, 5, 7)
    worldMap.placeElement(MapItem.FOODBIT, 2, 5)
    worldMap.placeElement(MapItem.FOODBIT, 6, 1)
    worldMap.placeElement(MapItem.FOODBIT, 1, 2)
    worldMap.placeElement(MapItem.FOODBIT, 2, 0)
    worldMap.placeElement(MapItem.FOODBIT, 0, 3)
    worldMap.placeElement(MapItem.FOODBIT, 3, 1)
    worldMap.placeElement(MapItem.FOODBIT, 1, 4)
    worldMap.placeElement(MapItem.FOODBIT, 4, 2)
    worldMap.placeElement(MapItem.FOODBIT, 5, 5)
    worldMap.placeElement(MapItem.FOODBIT, 7, 5)

    worldMap.placeElement(MapItem.FOODCLUSTER, 8, 7)

    worldMap.placeEater(agent)

    worldMap.render()

    while worldMap.checkEndConditionMet() == False:
        os.system('clear')
        nextAction = qlearner.QUseStep(agentXPose=agentX, agentYPos=agentY)

        curXpose = agentX
        curYpose = agentY

        # execute action
        if nextAction == 0:
            agentX -= 1
        elif nextAction == 1:
            agentX += 1
        elif nextAction == 2:
            agentY += 1
        elif nextAction == 3:
            agentY -= 1

        if agentX < 0:
            agentX = 0
        elif agentX >= gridSizeX:
            agentX = gridSizeX - 1

        if agentY < 0:
            agentY = 0
        elif agentY >= gridSizeY:
            agentY = gridSizeY - 1

        actionReward = worldMap.moveAgent(x=agentX, y=agentY)

        qlearner.updateQTable(xpos=curXpose, ypos=curYpose, xfuture=agentX, yfuture=agentY, action=nextAction, reward=actionReward)

        worldMap.render()
        time.sleep(1)
    