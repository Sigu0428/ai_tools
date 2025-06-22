import ludopy
import numpy as np
from ludoObs import LudoObs
from tqdm import tqdm
from rewards import Rewards
from log import Log
import random

class Trainer():
    def __init__(self):
        self.ludo = ludopy.Game()
        self.log = Log()

    def doEvalGame(self, players, rewards):
        ludo = ludopy.Game()
        tally = np.array([0, 0, 0, 0])
        returns = np.array([0, 0, 0, 0])
        rewards_buffer = [0, 0, 0, 0]
        reward_func = Rewards(rewards)
        prev_player_i = 0
        prev_obs = [None, None, None, None]
        t = 1
        while True:
                obs = LudoObs(ludo.get_observation()) #(dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner, there_is_a_winner, player_i)
                S = obs.getState()
                state_action_returns = players[obs.player_i].predictStateActionReturns(S, obs.move_pieces)
                A = players[obs.player_i].getAction(state_action_returns, obs.move_pieces, greedy = True) # get choice based on current network
                ludo.answer_observation(A)

                events = []
                _, prev_enemy_pieces = self.ludo.get_pieces(prev_player_i)
                rewards_buffer, events = reward_func.rewardTakePiece(prev_obs[prev_player_i], prev_enemy_pieces, rewards_buffer, events)
                rewards_buffer, events = reward_func.rewardGetPieceIntoPlay(obs, prev_obs[obs.player_i], rewards_buffer, events)
                rewards_buffer, events = reward_func.rewardLostPiece(obs, prev_obs[obs.player_i], rewards_buffer, events)
                rewards_buffer, events = reward_func.rewardCombPieceDist(obs, prev_obs[obs.player_i], rewards_buffer, events)
                rewards_buffer, events = reward_func.rewardPieceFinished(obs, prev_obs[obs.player_i], rewards_buffer, events)
                rewards_buffer, events = reward_func.rewardWinning(obs, rewards_buffer, events)

                returns[obs.player_i] = returns[obs.player_i] + (players[obs.player_i].hp['gamma']**t)*rewards_buffer[obs.player_i]
                rewards_buffer[obs.player_i] = 0
                
                prev_obs[obs.player_i] = obs
                prev_player_i = obs.player_i
                t += 1
                if obs.player_is_a_winner:
                    tally[obs.player_i] += 1
                    break
        return (tally, returns)

    def evaluate(self, n_samples, players, rewards):
        self.log.param_dict['evals'] = n_samples
        tally = np.array([0, 0, 0, 0])
        returns = []
        #for sample in range(n_samples):
        for sample in tqdm(range(n_samples)):
            random_idxs = np.random.choice(np.array([0, 1, 2, 3]), size=(4), replace=False)
            players_shuffled = [players[i] for i in random_idxs]
            (tally_shuffled, returns_shuffled) = self.doEvalGame(players_shuffled, rewards)
            for i, val in enumerate(tally_shuffled):
                tally[random_idxs[i]] += val
            returns_unshuffled = [0, 0, 0, 0]
            for i, val in enumerate(returns_shuffled):
                returns_unshuffled[random_idxs[i]] = val
            returns.append(returns_unshuffled[1])
        self.log.appendData("returns", np.mean(np.array(returns)))
        return tally
    
    def train(self, episodes, players, rewards, evals, num_evals=100):
        self.log.episodes_to_capture=np.linspace(0, episodes-1, num=10, dtype=int)
        self.log.param_dict['agent params'] = players[1].hp
        self.log.param_dict['episodes'] = episodes
        self.log.param_dict['rewards'] = rewards
        self.log.param_dict['continous_evals'] = evals
        self.log.param_dict['num continous_evals'] = num_evals
        reward_func = Rewards(rewards)
        winrate_capture_eps = np.linspace(0, episodes-1, num=num_evals, dtype=int)
        for ep in tqdm(range(episodes)):
            self.ludo.reset()
            prev_player_i = 0
            rewards_buffer = [0, 0, 0, 0]
            prev_obs = [None, None, None, None]
            turn = -1
            while True:
                turn += 1
                obs = LudoObs(self.ludo.get_observation()) #(dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner, there_is_a_winner, player_i)
                S = obs.getState()
                state_action_returns = players[obs.player_i].predictStateActionReturns(S, obs.move_pieces)
                A = players[obs.player_i].getAction(state_action_returns, obs.move_pieces) # get choice based on current network

                events = []
                #rewards_buffer[obs.player_i] += reward_func.ANY_MOVE
                _, prev_enemy_pieces = self.ludo.get_pieces(prev_player_i)
                rewards_buffer, events = reward_func.rewardTakePiece(prev_obs[prev_player_i], prev_enemy_pieces, rewards_buffer, events)
                rewards_buffer, events = reward_func.rewardGetPieceIntoPlay(obs, prev_obs[obs.player_i], rewards_buffer, events)
                rewards_buffer, events = reward_func.rewardLostPiece(obs, prev_obs[obs.player_i], rewards_buffer, events)
                rewards_buffer, events = reward_func.rewardCombPieceDist(obs, prev_obs[obs.player_i], rewards_buffer, events)
                rewards_buffer, events = reward_func.rewardPieceFinished(obs, prev_obs[obs.player_i], rewards_buffer, events)
                rewards_buffer, events = reward_func.rewardWinning(obs, rewards_buffer, events)
                self.log.appendInfo(ep=ep, turn=turn, obs=obs, info_dict={"state_action_returns": state_action_returns, "A": A, "rewards_buffer": rewards_buffer, "events": events})
                
                players[obs.player_i].OnEndOfTurn(S, A, rewards_buffer[obs.player_i], obs.move_pieces)
                rewards_buffer[obs.player_i] = 0
                self.ludo.answer_observation(A)
                
                prev_obs[obs.player_i] = obs
                prev_player_i = obs.player_i
                if obs.player_is_a_winner:
                    players[obs.player_i].OnEndOfTurn(S, A, rewards_buffer[obs.player_i], obs.move_pieces)
                    for player in players:
                        player.OnGameEnd()
                    break
            if any(ep == winr_ep for winr_ep in winrate_capture_eps):
                tally = self.evaluate(evals, players, rewards)
                winrate = tally[1]/evals
                self.log.appendData("winrate", winrate)
        return self.log