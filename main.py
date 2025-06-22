from trainer import Trainer
from randomWalkAgent import RandomWalkAgent
from tabularQlearner import tabularQLearner
from rewards import Rewards
from deepQlearner import deepQLearner
import keras
from priorityAgent import PriorityAgent
import numpy as np
from gradientBandit import GradientBandit

def gradientBandit():
    print("gradientBandit()")
    trainer = Trainer()
    hyper_parameters = {"alpha": 0.05,
                        "recent reward weight": 0.05,
                        "gamma": 0,
                        }
    players = [RandomWalkAgent(), GradientBandit(hyper_parameters, trainer.log), RandomWalkAgent(), RandomWalkAgent()]
    rewards = {"win": 0,
            "lose piece": -2.40,
            "take piece": 0.40,
            "hit goal": 0.10,
            "deploy": 0.80,
            "distance moved": 0,
            }
    log = trainer.train(5000, players, rewards)
    n_evals = 1000
    tally = trainer.evaluate(n_evals, players, rewards)
    winrate = tally[1]/n_evals
    log.dumpToFile("gradientBandit", winrate)
    print(winrate)

def epsilonExperiment():
    print("epsilonexperiment")
    for epsilon in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        trainer = Trainer()
        hyper_parameters = {"alpha": 0.01,
                            "gamma": 0.8,
                            "epsilon": epsilon,
                            "initial value": 0,
                            }
        players = [RandomWalkAgent(), tabularQLearner(hyper_parameters, trainer.log), RandomWalkAgent(), RandomWalkAgent()]
        rewards = {"win": 3200,
                "lose piece": -240,
                "take piece": 40,
                "hit goal": 10,
                "deploy": 80,
                "distance moved": 0,
                }
        log = trainer.train(2000, players, rewards, evals=100)
        log.dumpToFile("epsilon_exp_results", "epsilon" + str(epsilon))

def gammaExperiment():
    print("gammaexperiment")
    for gamma in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    #for gamma in [0.5, 0.6, 0.7, 0.8, 0.9]:
        trainer = Trainer()
        hyper_parameters = {"alpha": 0.01,
                            "gamma": gamma,
                            "epsilon": 1,
                            "initial value": 0,
                            }
        players = [RandomWalkAgent(), tabularQLearner(hyper_parameters, trainer.log), RandomWalkAgent(), RandomWalkAgent()]
        rewards = {"win": 3200,
                "lose piece": -240,
                "take piece": 40,
                "hit goal": 10,
                "deploy": 80,
                "distance moved": 0,
                }
        log = trainer.train(2000, players, rewards, evals=100)
        log.dumpToFile("gamma_exp_results", "gamma" + str(gamma))

def vsPriority():
    print("vs priority")
    trainer = Trainer()
    hyper_parameters = {"alpha": 0.01,
                        "gamma": 0.3,
                        "epsilon": 0.1,
                        "initial value": 0,
                        }
    players = [PriorityAgent(), tabularQLearner(hyper_parameters, trainer.log), PriorityAgent(), PriorityAgent()]
    rewards = {"win": 0,
               "lose piece": -240,
               "take piece": 40,
               "hit goal": 10,
               "deploy": 80,
               "distance moved": 0,
              }
    log = trainer.train(1000, players, rewards)
    n_evals = 1000
    tally = trainer.evaluate(n_evals, players, rewards)
    winrate = tally[1]/n_evals
    log.dumpToFile("tabularQLearning", winrate)
    print(winrate)

def main():
    # !!! recently changed the gamma around, 0.95 leads to too much noise, trying 0.3 now for tabular
    # !!! deep guy, halfing learning rate and doubling number of evaluations
    epsilonExperiment()

if __name__ == "__main__":
    main()