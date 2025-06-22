import random
import numpy as np
class GradientBandit:
    def __init__(self, hyperparameters, log):
        self.log = log
        self.hp = hyperparameters
        self.H = np.zeros((4, 5, 5, 5, 5, 5))
        self.R_avg = np.zeros((5, 5, 5, 5, 5))
        self.S_prev = None
        self.A_prev = None
        self.A_possible_prev = None
        self.t = 0
        self.G = 0

    def predictStateActionReturns(self, S, A_possible):
        preferences = []
        for A in A_possible:
            preferences.append(self.H[(A,) + tuple(S)])
        return preferences
    
    def policy(self, A_possible, preferences):
        policy = np.zeros((len(A_possible)))
        if len(A_possible) == 1:
            policy[0] = 1
            return policy
        for i in range(len(A_possible)):
            sum = 0
            for j in range(len(A_possible)):
                sum += np.exp(preferences[j])
            if sum == 0:
                if np.exp(preferences[i]) == 0:
                    policy = np.ones((len(A_possible)))*(1/len(A_possible)) #uniform policy
                    break
                else:
                    policy *= 0
                    policy[i] = 1
                    break
            policy[i] = np.exp(preferences[i])/sum
        for i in range(len(policy)):
            if np.isnan(policy[i]):
                print("pref", preferences)
                print("policy", policy)
                for i in range(len(A_possible)):
                    print(i)
                    sum = 0
                    for j in range(len(A_possible)):
                        sum += np.exp(preferences[j])
                        print(sum)
                    policy[i] = np.exp(preferences[i])/sum
                    print(policy)
        return policy

    def getAction(self, preferences, A_possible, greedy = False):
        if len(A_possible) == 0:
            return -1
        else:
            policy = self.policy(A_possible, preferences)
            A = np.random.choice(A_possible, p=policy)
        return A
    
    def OnGameEnd(self):
        self.log.appendData("returns", self.G)
        self.log.appendData("R_mean_means", np.mean(self.R_avg))
        self.t = 0
        self.G = 0
        self.S_prev = None
        self.A_prev = None

    def OnEndOfTurn(self, S_PRIME, A_PRIME, R, A_possible):
        #if self.S_prev is not None and R is not None:
        if self.S_prev is not None and len(self.A_possible_prev) > 0:
            preferences = self.predictStateActionReturns(self.S_prev, self.A_possible_prev)
            policy = self.policy(self.A_possible_prev, preferences)
            self.H[(self.A_prev,) + tuple(self.S_prev)] += self.hp['alpha']*(R - self.R_avg[tuple(self.S_prev)])*(1-policy[list(self.A_possible_prev).index(self.A_prev)])
            for a in self.A_possible_prev:
                if a != self.A_prev:
                    self.H[(a,) + tuple(self.S_prev)] += self.hp['alpha']*(R - self.R_avg[tuple(self.S_prev)])*policy[list(self.A_possible_prev).index(a)]
            self.R_avg[tuple(self.S_prev)] = (1 - self.hp['recent reward weight'])*self.R_avg[tuple(self.S_prev)] + self.hp['recent reward weight']*R
        
        self.G = self.G + (self.hp['gamma']**self.t)*R
        self.t += 1
        self.S_prev = S_PRIME
        self.A_prev = A_PRIME
        self.A_possible_prev = A_possible