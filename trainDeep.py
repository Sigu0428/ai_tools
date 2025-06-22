from trainer import Trainer
from randomWalkAgent import RandomWalkAgent
from deepQlearner import deepQLearner
import keras

def deepGuy():
    print("deepguy()")
    trainer = Trainer()
    hyper_parameters = {"prior_exp_window": 4000,
                        "batch_size": 64,
                        "learning_rate": 0.001,
                        "alpha": 0.001,
                        "gamma": 0.8,
                        "epochs": 10,
                        "explore_chance": 0.8,
                        "target_update_frequency": 100,
                        "training_frequency": 40,
                        #"NN_structure": [1024, 512, 256, 128, 64],
                        "NN_structure": [256, 256, 256],
                        "activation": keras.activations.relu,
                        "bootstrapping": True
                        }
    players = [RandomWalkAgent(), deepQLearner(hyper_parameters, trainer.log), RandomWalkAgent(), RandomWalkAgent()]
    rewards = {"win": 3200,
               "lose piece": -240,
               "take piece": 40,
               "hit goal": 10,
               "deploy": 80,
               "distance moved": 0,
              }
    log = trainer.train(2000, players, rewards, evals=100, num_evals=100)
    n_evals = 1000
    tally = trainer.evaluate(n_evals, players, rewards)
    winrate = tally[1]/n_evals
    log.dumpToFile("deepQLearning", winrate)
    print(winrate)

deepGuy()