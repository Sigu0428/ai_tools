from trainer import Trainer
from randomWalkAgent import RandomWalkAgent
from priorityAgent import PriorityAgent

def priorityGuy():
    print("priorityGuy()")
    trainer = Trainer()
    rewards = {"win": 0,
               "lose piece": -240,
               "take piece": 40,
               "hit goal": 10,
               "deploy": 80,
               "distance moved": 0,
              }
    players = [RandomWalkAgent(), PriorityAgent(), RandomWalkAgent(), RandomWalkAgent()]
    n_evals = 1000
    tally = trainer.evaluate(n_evals, players, rewards)
    winrate = tally[1]/n_evals
    print(winrate)

priorityGuy()