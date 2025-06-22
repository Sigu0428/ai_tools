from trainer import Trainer
from randomWalkAgent import RandomWalkAgent
from tabularQlearner import tabularQLearner

def tabularGuy():
    print("tabularGuy()")
    trainer = Trainer()
    hyper_parameters = {"alpha": 0.001,
                        "gamma": 0.8,
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
    log = trainer.train(2000, players, rewards, 100, 100)
    n_evals = 1000
    tally = trainer.evaluate(n_evals, players, rewards)
    winrate = tally[1]/n_evals
    log.dumpToFile("tabularQLearning", winrate)
    print(winrate)

tabularGuy()