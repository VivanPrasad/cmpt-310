# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#


from game import *
from learningAgents import ReinforcementAgent


import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        self.qValues = util.Counter() #dict
        self.alpha = args['alpha']
        self.epsilon = args['epsilon']
        self.discount = args['gamma']


    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
        """
        if (state, action) not in self.qValues:
            return 0.0
        return self.qValues[(state, action)]
        



    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        actions = self.getLegalActions(state)
        if not actions:
            return 0.0
        max_value = float('-inf')
        for action in actions:
            qValue = self.getQValue(state, action)
            if qValue > max_value: max_value = qValue
        return max_value


    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        actions = self.getLegalActions(state)
        if not actions:
            return None
        max_value = float('-inf')
        best_action = None
        for action in actions:
            qValue = self.getQValue(state, action)
            if qValue > max_value:
                max_value = qValue
                best_action = action
        return best_action


    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick legal action
        legalActions = self.getLegalActions(state)
        action = None
        if not legalActions:
            return action
        if util.flipCoin(self.epsilon): #using the util.flipCoin
            action = random.choice(legalActions)
        else:
            action = self.getPolicy(state)
        return action



    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        sample = reward + self.discount * self.computeValueFromQValues(nextState)
        self.qValues[(state, action)] = (1-self.alpha) *self.qValues[(state, action)] + (self.alpha * sample)
        """
              QLearning update algorithm:

              Q(s,a) = (1-alpha)*Q(s,a) + alpha*sample

              ***sample = R(s,a,s') + gamma*max(Q(s',a'))***

        """



    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)

