import random
class PriorityAgent:
    def __init__(self, epsilon=0.01, gamma = 0.5, alpha=0.1) -> None:
        self.epsilon = epsilon
        self.S_prev = None
        self.A_prev = None
        self.gamma = gamma
        self.alpha = alpha
        self.hp = {"gamma": 0}
    
    def OnGameEnd(self):
        pass

    def OnEndOfTurn(self, S_PRIME, A_PRIME, R, A_possible):
        pass

    def predictStateActionReturns(self, S, A_possible):
        return S
    
    def getAction(self, state_action_returns, A_possible, greedy = False):
        S = state_action_returns
        #S = (self.whichCanTake(), self.whichCanDeploy(), self.whichCanEscapeDanger(), self.whichCanHitGoal(), self.whichWontDie())
        priorities = [1, 2, 0, 3, 4]
        A = 4
        for idx in priorities:
            if A == 4: A = S[idx]
        
        if len(A_possible) == 0:
            return -1
        if not any(A == a for a in A_possible):
            #return random.choice(A_possible)
            return A_possible[0]
        return A