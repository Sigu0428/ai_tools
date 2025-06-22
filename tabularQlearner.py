import random
import numpy as np
class tabularQLearner:
    def __init__(self, hyperparameters, log):
        self.log = log
        self.hp = hyperparameters
        self.Q = np.ones((4, 5, 5, 5, 5, 5))*self.hp['initial value']
        self.S_prev = None
        self.A_prev = None
        self.t = 0
        self.G = 0

    def predictStateActionReturns(self, S, A_possible):
        returns = []
        for A in A_possible:
            returns.append(self.Q[(A,) + tuple(S)])
        return (returns, S)
    
    def getAction(self, state_action_returns, A_possible, greedy = False):
        state_action_returns, S = state_action_returns
        if len(A_possible) == 0:
            return -1
        if (random.random() < self.hp['epsilon']) and not greedy:
            A = random.choice(A_possible)
        else:
            # maximum with random tie-breaking
            A = A_possible[np.random.choice(np.flatnonzero(np.isclose(np.array(state_action_returns), np.array(state_action_returns).max())))]
        return A
    
    def OnGameEnd(self):
        self.log.appendData("returns", self.G)
        self.log.appendData("Q_means", np.mean(self.Q))
        self.t = 0
        self.G = 0
        self.S_prev = None
        self.A_prev = None

    def OnEndOfTurn(self, S_PRIME, A_PRIME, R, A_possible):
        #if self.S_prev is not None and R is not None:
        if self.S_prev is not None and len(A_possible) > 0:
            max_Q = self.Q[(A_possible[0], ) + tuple(S_PRIME)]
            for a in A_possible:
                max_Q = max((max_Q, self.Q[(a, ) + tuple(S_PRIME)]))
            #delta = self.alpha*(R + self.gamma*Q[(A_PRIME,) + S_PRIME] - Q[(self.A_prev,) + self.S_prev])
            delta = self.hp['alpha']*(R + self.hp['gamma']*max_Q - self.Q[(self.A_prev,) + tuple(self.S_prev)])
            self.Q[(self.A_prev,) + tuple(self.S_prev)] += + delta
            #print(self.alpha*(R + self.gamma*Q[(A_PRIME,) + S_PRIME] - Q[(self.A_prev,) + self.S_prev]))
            #print("self.S_prev", self.S_prev, "S_PRIME", S_PRIME, "self.A_prev", self.A_prev, "A_PRIME", A_PRIME, "R", R, "Q[prev] +=", delta, "Q[prev]", Q[(self.A_prev,) + self.S_prev], "Q[prime]", Q[(A_PRIME,) + S_PRIME])
        
        self.G = self.G + (self.hp['gamma']**self.t)*R
        self.t += 1
        self.S_prev = S_PRIME
        self.A_prev = A_PRIME
        return self.Q