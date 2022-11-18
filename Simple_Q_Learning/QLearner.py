import numpy as np

class QLearner:
    def __init__(self, gridXSize, gridYSize, numActions, discountRate, learnRate):
        self.learnRate = learnRate
        self.discountRate = discountRate
        self.epsilon = 1
        # Actions:
        #   Left = 0
        #   Right = 1
        #   Up = 2
        #   Down = 3
        self.actionSpaceSize = numActions
        self.observationSpaceSize = gridXSize * gridYSize
        self.Qtable = np.zeros((self.observationSpaceSize, self.actionSpaceSize))
        self.outOfBoundsPenalty = -10000

        for i in range(0, gridXSize):
            self.Qtable[i*10, 3] = self.outOfBoundsPenalty
            self.Qtable[10*i+gridYSize-1, 2] = self.outOfBoundsPenalty

        for j in range(0, gridYSize):
            self.Qtable[j, 0] = self.outOfBoundsPenalty
            self.Qtable[j+10*(gridXSize - 1), 0] = self.outOfBoundsPenalty

    def updateQTable(self, xpos, ypos, xfuture, yfuture, action, reward):
        qind = 10 * xpos + ypos
        qfuture = 10 * xfuture + yfuture
        newQ = self.calculateQ(action=action, reward=reward, qind=qind, qfuture=qfuture)
        self.Qtable[qind, action] = newQ

    def calculateQ(self, action, reward, qind, qfuture):
        maxR = self.Qtable[qfuture, :].max()
        newQ = (1 - self.learnRate) * self.Qtable[qind, action] + self.learnRate * (reward + (self.discountRate * maxR))
        return newQ

    def QLearnStep(self, agentXPose, agentYPose):
        qind = 10 * agentXPose + agentYPose
        exploitProb = np.random.rand()

        if exploitProb > self.epsilon:
            maxInd = np.argmax(self.Qtable[qind, :])
            nextAction = int(maxInd)
            return nextAction
        
        nextAction = int(np.random.randint(4))

        if agentXPose <= 0:
            nextAction = int(np.random.randint(1, 4))
        elif agentXPose >= 9:
            nextAction = int(np.random.choice([0, np.random.randint(2,4)]))

        if agentYPose <= 0:
            nextAction = int(np.random.randint(3))
        elif agentYPose >= 9:
            nextAction = int(np.random.choice([3, np.random.randint(2)]))

        if agentXPose <= 0 and agentYPose <= 0:
            nextAction = int(np.random.randint(1, 3))
        elif agentXPose <= 0 and agentYPose >= 9:
            nextAction = int(np.random.choice([1, 3]))
        elif agentXPose >= 9 and agentYPose >= 9:
            nextAction = int(np.random.choice([0, 3]))
        elif agentXPose >= 9 and agentYPose <= 0:
            nextAction = int(np.random.choice([1, 3]))

        # Update epsilon for greedy strategy
        if self.Qtable[qind, nextAction] == 0:
            self.epsilon -= 1 / (self.actionSpaceSize * self.observationSpaceSize)

        return nextAction

    def QUseStep(self, agentXPose, agentYPos):
        qind = 10 * agentXPose + agentYPos

        maxInd = np.argmax(self.Qtable[qind, :])
        nextAction = int(maxInd)
        return nextAction


    def setQTable(self, newQTable):
        self.Qtable = newQTable
