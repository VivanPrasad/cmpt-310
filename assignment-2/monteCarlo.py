
import copy
import random
import time
import sys
import math
from collections import namedtuple

GameState = namedtuple('GameState', 'to_move, move, utility, board, moves')

def random_player(game, state):
    """A random player that chooses a legal move at random."""
    return random.choice(game.actions(state)) if game.actions(state) else None

# MonteCarlo Tree Search support

class MCTS: #Monte Carlo Tree Search implementation
    class Node:
        def __init__(self, state, par=None):
            self.state = copy.deepcopy(state)

            self.parent = par
            self.children = []
            self.visitCount = 0
            self.winScore = 0

        def getChildWithMaxScore(self):
            maxScoreChild = max(self.children, key=lambda x: x.visitCount)
            return maxScoreChild



    def __init__(self, game, state):
        self.root = self.Node(state)
        self.state = state
        self.game = game
        self.exploreFactor = math.sqrt(2)

    def isTerminalState(self, utility, moves):
        return utility != 0 or len(moves) == 0
        
    def monteCarloPlayer(self, timelimit = 4):
        """Entry point for Monte Carlo tree search"""
        start = time.perf_counter()
        end = start + timelimit
        """
        Use time.perf_counter() to apply iterative deepening strategy.
         At each iteration we perform 4 stages of MCTS: 
         SELECT, EXPEND, SIMULATE, and BACKUP. Once time is up
        we use getChildWithMaxScore() to pick the node to move to
        """
        #print("MCTS: your code goes here. 10pt.")
        while time.perf_counter() < end:
            self.selectNode(self.root)
            self.expandNode(self.root)
            if time.perf_counter() >= end: break
            winner = self.simulateRandomPlay(self.root)
            self.backPropagation(self.root, winner)

        winner_node = self.root.getChildWithMaxScore()
        assert(winner_node is not None)
        return winner_node.state.move


    """SELECT stage function. walks down the tree using findBestNodeWithUCT()"""
    def selectNode(self, nd: Node):
        node = nd
        #print("Your code goes here 5pt.")
        while node.children:
            node = self.findBestNodeWithUCT(node)
            if node is None:
                break
        return node

    def findBestNodeWithUCT(self, nd):
        """finds the child node with the highest UCT. Parse nd's children and use uctValue() to collect uct's for the
        children....."""
        childUCT = []
        #print("Your code goes here 2pt.")
        for child in nd.children:
            childUCT.append((child, self.uctValue(nd.visitCount, child.winScore, child.visitCount)))
        return max(childUCT, key=lambda x: x[1])[0] if childUCT else None

    def uctValue(self, parentVisit, nodeScore, nodeVisit):
        """compute Upper Confidence Value for a node"""
        # print("Your code goes here 3pt.")
        if nodeVisit == 0:
            return float('inf')  # Explore unvisited nodes first
        exploration_term = self.exploreFactor * math.sqrt(math.log(parentVisit) / nodeVisit)
        return (nodeScore / nodeVisit) + exploration_term

    """EXPAND stage function. """
    def expandNode(self, nd):
        """generate all the possible child nodes and append them to nd's children"""
        stat = nd.state
        tempState = GameState(to_move=stat.to_move, move=stat.move, utility=stat.utility, board=stat.board, moves=stat.moves)
        for a in self.game.actions(tempState):
            childNode = self.Node(self.game.result(tempState, a), nd)
            nd.children.append(childNode)

    """SIMULATE stage function"""
    def simulateRandomPlay(self, nd):
        # first use compute_utility() to check win possibility for the current node. IF so, return the winner's symbol X, O or N representing tie
        winStatus = self.game.compute_utility(nd.state.board, nd.state.move, nd.state.board[nd.state.move])
        #print("Your code goes here 5pt.")

        """now roll out a random play down to a terminating state. """

        tempState = copy.deepcopy(nd.state) # to be used in the following random playout
        to_move = tempState.to_move
        #print("Your code goes here 5pt.")
        while not self.isTerminalState(winStatus, tempState.moves):
            moves = self.game.actions(tempState)
            a = random.choice(moves) if moves else None
            if a is not None:
                tempState = self.game.result(tempState, a)
                to_move = self.game.to_move(tempState)
            winStatus = self.game.compute_utility(tempState.board, tempState.move, tempState.board[tempState.move])

        return ('X' if winStatus > 0 else 'O' if winStatus < 0 else 'N') # 'N' means tie


    def backPropagation(self, nd, winningPlayer):
        """propagate upword to update score and visit count from
        the current leaf node to the root node."""
        tempNode = nd
        #print("Your code goes here 5pt.")
        while tempNode is not None:
            tempNode.visitCount = tempNode.visitCount + 1
            if winningPlayer == tempNode.state.to_move:
                tempNode.winScore = tempNode.winScore + 1
            tempNode = tempNode.parent