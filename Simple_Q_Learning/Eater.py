
class Eater:
    def __init__(self):
        self.stomach = 0
        self.lifespan = 100

    def updateReward(self, reVal):
        self.stomach += reVal

    def decreaseLifeSpan(self):
        self.lifespan -= 1

    def getTotalReward(self):
        return self.stomach

    def isDead(self):
        if self.lifespan <= 0:
            return True
        else:
            return False
