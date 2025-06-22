import numpy as np
class Rewards():
    def __init__(self, rewards):
        self.rewards = rewards

    class FIELDS():
        # Ludo Fields
        GLOBES = (1, 9, 22, 35, 48)
        ENEMY_GLOBES = (9, 14, 22, 27, 35, 40, 48)
        STARS = (5, 12, 18, 25, 38, 44, 51)
        GOAL_STRETCH_START = 53
        GOAL = 59
        HOME = 0

    def rewardGetPieceIntoPlay(self, obs, prev_obs, rewards, events):
        # check if current player got knocked home since it last had a turn
        if prev_obs is not None:
            for i in range(obs.player_pieces.shape[0]):
                if obs.player_pieces[i] != 0 and prev_obs.player_pieces[i] == 0:
                    rewards[obs.player_i] += self.rewards['deploy']
                    events.append(obs.getColorOfPlayer(prev_obs.player_i) + " got a piece out of home since its last turn")
        return (rewards, events)
    
    def rewardLostPiece(self, obs, prev_obs, rewards, events):
        # check if current player got knocked home since it last had a turn
        if prev_obs is not None:
            for i in range(obs.player_pieces.shape[0]):
                if obs.player_pieces[i] == 0 and prev_obs.player_pieces[i] != 0:
                    rewards[obs.player_i] += self.rewards['lose piece']
                    events.append(obs.getColorOfPlayer(prev_obs.player_i) + " got a piece knocked home since its last turn")
        return (rewards, events)

    def rewardTakePiece(self, prev_player_obs, prev_players_current_enemies, rewards, events):
        # check if previous player knocked someone home
        if prev_player_obs is not None:
            prev_players_prev_enemy_pieces = prev_player_obs.enemy_pieces
            for i in range(len(prev_players_current_enemies)):
                for j in range(len(prev_players_current_enemies[i])):
                    if prev_players_current_enemies[i][j] == 0 and prev_players_prev_enemy_pieces[i][j] != 0:
                        rewards[prev_player_obs.player_i] += self.rewards['take piece']
                        events.append(prev_player_obs.getColorOfPlayer(prev_player_obs.player_i) + " took a piece from " + prev_player_obs.getColorOfPlayer((i + prev_player_obs.player_i + 1)%4))
        return (rewards, events)

    def rewardWinning(self, obs, rewards, events):
        # check if player has won the game
        if obs.player_is_a_winner:
            rewards[obs.player_i] += self.rewards['win']
            events.append(obs.getColorOfPlayer(obs.player_i) + " won the game")
        return (rewards, events)

    def rewardPieceFinished(self, obs, prev_obs, rewards, events):
        #check if player got a piece home since its last turn
        if prev_obs is not None:
            for i in range(obs.player_pieces.shape[0]):
                if obs.player_pieces[i] == self.FIELDS.GOAL and prev_obs.player_pieces[i] != self.FIELDS.GOAL:
                    rewards[obs.player_i] += self.rewards['hit goal'] 
                    events.append(obs.getColorOfPlayer(obs.player_i) + " got a piece finished")
        return (rewards, events)
    
    def rewardCombPieceDist(self, obs, prev_obs, rewards, events):
        #reward player based on how far (positive or negative) its pieces have moved since last turn
        if prev_obs is not None:
            reward = (np.sum(obs.player_pieces) - np.sum(prev_obs.player_pieces))*self.rewards['distance moved']
            rewards[obs.player_i] += reward
            events.append(obs.getColorOfPlayer(obs.player_i) + " got " + str(reward) + " for movement since last turn")
        return (rewards, events)