#######################################################
# Player created
#
# Swansea University
# Mar 2023
#
# Gonzalo M Flores (2026765), work by my own
# 
# This file is created inside a folder called MyPlayerAgent
# The user can play using the next commands like: 
#       python gomoku.py MyPlayerAgent MyPlayerAgent
#       python gomoku.py GomokuAgentRand GomokuAgentRand
#       python gomoku.py GomokuAgentRand MyPlayerAgent
#######################################################

import numpy as np
from gomokuAgent import GomokuAgent
from misc import legalMove, winningTest


class Player(GomokuAgent):
    def move(self, board):
        depth = 3  # You can adjust the depth to control the search depth of the algorithm
        # Call the minimax_alpha_beta function to find the best move
        move = self.minimax_alpha_beta(board, depth, -np.inf, np.inf, True)
        return move[1]

    def minimax_alpha_beta(self, board, depth, alpha, beta, maximizing_player):
        # Base case: if depth is 0 or there is a winner, return the evaluation score and move
        if depth == 0 or winningTest(self.ID, board, self.X_IN_A_LINE) or winningTest(-self.ID, board, self.X_IN_A_LINE):
            return self.evaluate_board(board), None

        # If it is the maximizing players turn, find the best move with the highest evaluation value
        if maximizing_player:
            max_eval = -np.inf
            best_move = None
            for move in self.get_legal_moves(board):
                new_board = board.copy()
                new_board[move] = self.ID
                eval_value = self.minimax_alpha_beta(
                    new_board, depth - 1, alpha, beta, False)[0]
                if eval_value > max_eval:
                    max_eval = eval_value
                    best_move = move
                alpha = max(alpha, eval_value)
                if beta <= alpha:
                    break
            return max_eval, best_move
        
        # If it is the minimizing players turn, find the best move with the lowest evaluation value
        else:
            min_eval = np.inf
            best_move = None
            for move in self.get_legal_moves(board):
                new_board = board.copy()
                new_board[move] = -self.ID
                eval_value = self.minimax_alpha_beta(
                    new_board, depth - 1, alpha, beta, True)[0]
                if eval_value < min_eval:
                    min_eval = eval_value
                    best_move = move
                beta = min(beta, eval_value)
                if beta <= alpha:
                    break
            return min_eval, best_move

    def get_legal_moves(self, board):
        # Find all legal moves on the board
        legal_moves = []
        for r in range(self.BOARD_SIZE):
            for c in range(self.BOARD_SIZE):
                if board[r, c] == 0:
                    legal_moves.append((r, c))
        return legal_moves

    def evaluate_board(self, board):
        # I have chosen to use a random evaluation for simplicity, but a custom evaluation function
        # considering factors like the number of stones in a row, open ends, etc. can be implemented
        return np.random.random()