import numpy as np
class LudoObs():
    GLOBES = [[(6, 13), (2, 8), (6, 2), (12, 6), (8, 12)],
              [(1, 6), (6, 2), (12, 6), (8, 12), (2, 8)],
              [(8, 1), (12, 6), (8, 12), (2, 8), (6, 2)],
              [(13, 8), (8, 12), (2, 8), (6, 2), (12, 6)]
    ]
    STARS = [(6, 9), (0, 7), (5, 6), (7, 0), (8, 5), (14, 7), (9, 8), (7, 14)]
    GOAL = 57
    STAR_BEFORE_GOAL = 51
    HOME = 0
    HOME_STRECH = 52
    
    def __init__(self, ludo_obs):
        (self.dice, self.move_pieces, self.player_pieces, self.enemy_pieces, self.player_is_a_winner, self.there_is_a_winner), self.player_i = ludo_obs
    
    def posEquals(self, pos1, pos2):
        return all((pos1[0] == pos2[0], pos1[1] == pos2[1]))
    
    def enemyAtPos(self, pos):
        num = 0
        enemy_idx = 0
        enemy_idxs = [(self.player_i+1)%4, (self.player_i+2)%4, (self.player_i+3)%4] # fr fr
        for i, enemy in enumerate(self.enemy_pieces):
            idx = enemy_idxs[i]
            for piece in enemy:
                if self.posEquals(self.path[idx][piece], pos):
                    num += 1
                    enemy_idx = idx
        return (enemy_idx, num)
    
    def vulnerablePieceAtPos(self, pos):
        if pos >= self.HOME_STRECH: return False
        new_2Dpos = self.path[self.player_i][pos]
        (enemy_idx, num_enemies) = self.enemyAtPos(new_2Dpos)
        is_globe = any(self.posEquals(new_2Dpos, globe) for globe in self.GLOBES[enemy_idx])
        if num_enemies == 1 and not is_globe: return True
        if pos == 1 and num_enemies > 0: return True # pieces that deploy are unstopable
        return False

    def whichCanTake(self): #does not account for stars
        for idx in self.move_pieces:
            new_pos = 1 if self.canDeploy(idx) else self.player_pieces[idx] + self.dice
            if self.vulnerablePieceAtPos(new_pos): return idx
        return 4 # no piece can take

    def whichCanHitGoal(self):
        for idx in self.move_pieces:
            new_pos = self.player_pieces[idx] + self.dice
            if new_pos == self.GOAL: return idx
            if new_pos == self.STAR_BEFORE_GOAL: return idx
        return 4 # no piece can hit goal
    
    def wontDie(self, idx):
        pos = self.player_pieces[idx]
        new_pos = pos + self.dice
        if new_pos >= self.HOME_STRECH: return True
        if self.canDeploy(idx): return True
        new_2Dpos = self.path[self.player_i][new_pos]
        (enemy_idx, num_enemies) = self.enemyAtPos(new_2Dpos)
        is_enemy_globe = any(self.posEquals(new_2Dpos, globe) for globe in self.GLOBES[enemy_idx])
        if num_enemies < 1 and not is_enemy_globe: return True
                
    def whichWontDie(self):
        for idx in self.move_pieces:
            if self.wontDie(idx): return idx
        return 4 # all pieces will go home

    def canDeploy(self, idx):
        return self.player_pieces[idx] == self.HOME and self.dice == 6
    
    def whichCanDeploy(self):
        for i in range(len(self.player_pieces)):
            if self.canDeploy(i): return i
        return 4 # no piece can deploy
    
    def isInDanger(self, idx, pos):
        if pos >= self.HOME_STRECH: return False
        if pos == self.HOME: return False
        pos2D = self.path[self.player_i][pos]
        enemy_idxs = [(self.player_i+1)%4, (self.player_i+2)%4, (self.player_i+3)%4]
        for i, enemy_idx in enumerate(enemy_idxs):
            if self.posEquals(pos2D, self.path[enemy_idx][1]) and any(piece == 0 for piece in self.enemy_pieces[i]): return True
        on_globe = any(self.posEquals(pos2D, globe) for globe in self.GLOBES[self.player_i])
        is_stacked = any((pos == other_piece) and j != idx for j, other_piece in enumerate(self.player_pieces))
        if on_globe: return False
        if is_stacked: return False
        for j in range(1, 7):
            new_pos = pos - j
            if new_pos < 1: new_pos += 52 # this index in the players path corresponds to the point right before the start (because ludopy doesnt understand the rules of ludo)
            (enemy_idx, num_enemies) = self.enemyAtPos(self.path[self.player_i][new_pos])
            if num_enemies > 0:
                return True
        return None
        
    def whichCanEscapeDanger(self):
        for idx in self.move_pieces:
            pos = self.player_pieces[idx]
            if self.canDeploy(idx): continue
            new_pos = pos + self.dice
            if self.isInDanger(pos, idx) and not self.isInDanger(new_pos, idx): return idx
        return 4 # no pieces can escape danger
                
    def getState(self):
        if True:
            S = np.array((self.whichCanTake(), self.whichCanDeploy(), self.whichCanEscapeDanger(), self.whichCanHitGoal(), self.whichWontDie()))
        if False:
            S = np.concatenate((self.player_pieces, self.enemy_pieces.flatten(), np.array([self.dice])))
        return S
    
    ludoAsciiBoard = '''                  ┌──┬─S┌──┐                  
                  │  │  │  │                  
                  ├──┼──┼──┤  ┌──┐  ┌──┐      
                  │  │  │  G  │  │  │  │      
   ┌──┐  ┌──┐     ├──┼──┼──┤  └──┘  └──┘      
   │  │  │  │     G  │  │  │                  
   └──┘  └──┘     ├──┼──┼──┤  ┌──┐  ┌──┐      
                  │  │  │  │  │  │  │  │      
   ┌──┐  ┌──┐     ├──┼──┼──┤  └──┘  └──┘      
   │  │  │  │     │  │  │  │                  
   └──┘  └──┘     ├──┼──┼──┤                  
                  │  │  │  S                  
┌──┬─G┌──┬──┬──┬─S├──┴──┴──┼──┬──┬──┬─G┌──┬──┐
│  │  │  │  │  │  │        │  │  │  │  │  │  │
├──┼──┼──┼──┼──┼──┤        ├──┼──┼──┼──┼──┼──┤
S  │  │  │  │  │  │        │  │  │  │  │  │  S
├──┼──┼──┼──┼──┼──┤        ├──┼──┼──┼──┼──┼──┤
│  │  │  │  │  │  │        │  │  │  │  │  │  │
└──┴──┴─G└──┴──┴──┼──┬──┬──┼─S└──┴──┴──┴─G└──┘
                  S  │  │  │                  
                  ├──┼──┼──┤     ┌──┐  ┌──┐   
                  │  │  │  │     │  │  │  │   
      ┌──┐  ┌──┐  ├──┼──┼──┤     └──┘  └──┘   
      │  │  │  │  │  │  │  │                  
      └──┘  └──┘  ├──┼──┼──┤     ┌──┐  ┌──┐   
                  │  │  │  G     │  │  │  │   
      ┌──┐  ┌──┐  ├──┼──┼──┤     └──┘  └──┘   
      │  │  │  │  G  │  │  │                  
      └──┘  └──┘  ├──┼──┼──┤                  
                  │  │  │  │                  
                  └──┴─S└──┘                  '''

    colors = ['g', 'y', 'b', 'r']
    path = [
        [(4, 13), (6, 13), (6, 12), (6, 11), (6, 10), (6, 9), (5, 8), (4, 8), (3, 8), (2, 8), (1, 8), (0, 8), (0, 7), (0, 6), (1, 6), (2, 6), (3, 6), (4, 6), (5, 6), (6, 5), (6, 4), (6, 3), (6, 2), (6, 1), (6, 0), (7, 0), (8, 0), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (9, 6), (10, 6), (11, 6), (12, 6), (13, 6), (14, 6), (14, 7), (14, 8), (13, 8), (12, 8), (11, 8), (10, 8), (9, 8), (8, 9), (8, 10), (8, 11), (8, 12), (8, 13), (8, 14), (7, 14), (7, 13), (7, 12), (7, 11), (7, 10), (7, 9), (7, 8)],
        [(1, 4), (1, 6), (2, 6), (3, 6), (4, 6), (5, 6), (6, 5), (6, 4), (6, 3), (6, 2), (6, 1), (6, 0), (7, 0), (8, 0), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (9, 6), (10, 6), (11, 6), (12, 6), (13, 6), (14, 6), (14, 7), (14, 8), (13, 8), (12, 8), (11, 8), (10, 8), (9, 8), (8, 9), (8, 10), (8, 11), (8, 12), (8, 13), (8, 14), (7, 14), (6, 14), (6, 13), (6, 12), (6, 11), (6, 10), (6, 9), (5, 8), (4, 8), (3, 8), (2, 8), (1, 8), (0, 8), (0, 7), (1, 7), (2, 7), (3, 7), (4, 7), (5, 7), (6, 7)],
        [(10, 1), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (9, 6), (10, 6), (11, 6), (12, 6), (13, 6), (14, 6), (14, 7), (14, 8), (13, 8), (12, 8), (11, 8), (10, 8), (9, 8), (8, 9), (8, 10), (8, 11), (8, 12), (8, 13), (8, 14), (7, 14), (6, 14), (6, 13), (6, 12), (6, 11), (6, 10), (6, 9), (5, 8), (4, 8), (3, 8), (2, 8), (1, 8), (0, 8), (0, 7), (0, 6), (1, 6), (2, 6), (3, 6), (4, 6), (5, 6), (6, 5), (6, 4), (6, 3), (6, 2), (6, 1), (6, 0), (7, 0), (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6)],
        [(13, 10), (13, 8), (12, 8), (11, 8), (10, 8), (9, 8), (8, 9), (8, 10), (8, 11), (8, 12), (8, 13), (8, 14), (7, 14), (6, 14), (6, 13), (6, 12), (6, 11), (6, 10), (6, 9), (5, 8), (4, 8), (3, 8), (2, 8), (1, 8), (0, 8), (0, 7), (0, 6), (1, 6), (2, 6), (3, 6), (4, 6), (5, 6), (6, 5), (6, 4), (6, 3), (6, 2), (6, 1), (6, 0), (7, 0), (8, 0), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (9, 6), (10, 6), (11, 6), (12, 6), (13, 6), (14, 6), (14, 7), (13, 7), (12, 7), (11, 7), (10, 7), (9, 7), (8, 7)],
        ]
    
    @staticmethod
    def getColorOfPlayer(player_i):
        return LudoObs.colors[player_i]
    
    @staticmethod
    def toStringIdx(x, y):
        return (y*2 + 1)*47 + x*3 + 1
    
    @staticmethod
    def placeChar(board, x, y, ch):
        idx = LudoObs.toStringIdx(x, y)
        board = board[:idx] + ch + board[idx+len(ch):]
        return board
    
    def getBoard(self):
        board = self.ludoAsciiBoard
        
        player_idx = self.player_i
        player_pieces = self.player_pieces
        for piece in player_pieces:
            x, y = self.path[player_idx][piece]
            existing_char = board[LudoObs.toStringIdx(x, y)]
            if existing_char == ' ':
                board = LudoObs.placeChar(board, x, y, '1' + self.colors[player_idx])
            else:
                board = LudoObs.placeChar(board, x, y, str(eval(existing_char)+1) + self.colors[player_idx])
        
        enemy_pieces = self.enemy_pieces
        enemy_idxs = [(player_idx+1)%4, (player_idx+2)%4, (player_idx+3)%4] # fr fr
        for i, enemy in enumerate(enemy_pieces):
            idx = enemy_idxs[i]
            for piece in enemy:
                x, y = self.path[idx][piece]
                existing_char = board[LudoObs.toStringIdx(x, y)]
                if existing_char == ' ':
                    board = LudoObs.placeChar(board, x, y, '1' + self.colors[idx])
                else:
                    board = LudoObs.placeChar(board, x, y, str(eval(existing_char)+1) + self.colors[idx])

        board = LudoObs.placeChar(board, 9, 14, self.colors[player_idx] + "'s turn: dice=" + str(self.dice))
        return board