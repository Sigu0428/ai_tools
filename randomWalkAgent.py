import random
class RandomWalkAgent:
    def __init__(self, epsilon=0.01, gamma = 0.5, alpha=0.1) -> None:
        self.epsilon = epsilon
        self.S_prev = None
        self.A_prev = None
        self.gamma = gamma
        self.alpha = alpha
        self.hp = {"gamma": 0}

    def predictStateActionReturns(self, S, A_possible):
        return None
    
    def getAction(self, state_action_returns, A_possible, greedy = False):
        if len(A_possible) == 0:
            return -1
        A = random.choice(A_possible)
        return A
    
    def OnGameEnd(self):
        pass

    def OnEndOfTurn(self, S_PRIME, A_PRIME, R, A_possible):
        pass