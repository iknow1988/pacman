# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from captureAgents import CaptureAgent
import random
import time
import util
from game import Directions
import game
import distanceCalculator
import random
import time
import util
import sys
import numpy as np
import math
from util import nearestPoint
import sys
import os
sys.path.append('teams/poison/')
import pickle

random.seed(42)
#################
# Team creation #
#################


def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveUCT', second='DefensiveQAgent'):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """

    # we need to reference here our agents for pacman / ghosts
    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

# experimental test with baseline


class TestReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that chooses score-maximizing actions
    """

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = gameState.getLegalActions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(gameState, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        foodLeft = len(self.getFood(gameState).asList())

        if foodLeft <= 2:
            bestDist = 9999
            for action in actions:
                successor = self.getSuccessor(gameState, action)
                pos2 = successor.getAgentPosition(self.index)
                dist = self.getMazeDistance(self.start, pos2)
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist
            return bestAction

        return random.choice(bestActions)

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def getFeatures(self, gameState, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)
        return features

    def getWeights(self, gameState, action):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        return {'successorScore': 1.0}


class TestOffensiveReflexAgent(TestReflexCaptureAgent):
    """
    A reflex agent that seeks food. This is an agent
    we give you to get an idea of what an offensive agent might look like,
    but it is by no means the best or only way to build an offensive agent.
    """

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        foodList = self.getFood(successor).asList()
        features['successorScore'] = -len(foodList)  # self.getScore(successor)

        # Compute distance to the nearest food

        if len(foodList) > 0:  # This should always be True,  but better safe than sorry
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food)
                               for food in foodList])
            features['distanceToFood'] = minDistance
        return features

    def getWeights(self, gameState, action):
        return {'successorScore': 100, 'distanceToFood': -1}

    def getRemainingScareTime(self, gameState, agentIndex):
        return gameState.getAgentState(agentIndex).scaredTimer


class TestDefensiveReflexAgent(TestReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if myState.isPacman:
            features['onDefense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i)
                   for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition()
                    != None]
        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.getMazeDistance(
                myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        missingfoodinf = getMissingFood(self, gameState)
        # print(missingfoodinf)
        dist_miss = 0
        features['MissingFood'] = 0
        if len(missingfoodinf) > 0:
            # Weight should be modified
            features['MissingFood'] = 5
            # Try to use distance to measure what action should be taken
            for pos, i in missingfoodinf:
                dist_miss = self.getMazeDistance(pos, myPos)
        # print(5*dist_miss)
        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if myState.isPacman:
            features['onDefense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i)
                   for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition()
                    != None]
        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.getMazeDistance(
                myPos, a.getPosition()) for a in invaders]
            # print dists
            features['invaderDistance'] = min(dists)

        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(
            self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        return features

    def getWeights(self, gameState, action):
        return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}


# experimental test with MCT
class MCTBasedAgent(CaptureAgent):
    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        # print "Features",features
        # print "Weights",weights
        # print "Eval",features * weights
        return features * weights

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)
        return features

    def getWeights(self, gameState, action):
        return {'successorScore': 1.0}


class OffensiveMCT(MCTBasedAgent):

    def getFeatures(self, gameState, action):
        """
        Get features used for state evaluation.
        """
        features = util.Counter()

        successor = self.getSuccessor(gameState, action)
        # Compute score from successor state
        features['successorScore'] = self.getScore(gameState)

        if self.food_carrying >= self.MAX_FOOD_CARRYING or len(self.getFood(gameState).asList()) <= 2:
            features['successorScore'] = self.getScore(gameState)
        else:
            features['successorScore'] = -len(self.getFood(gameState).asList())

        foodList = self.getFood(successor).asList()
        """
        # compute distance to the nearest cluster of food : Not working properly, Pacman is getting stucked in some positions
        if len(foodList) > 0 and self.food_carrying <self.MAX_FOOD_CARRYING:
          successor_food_clusters=kmeans(self.getFood(successor),3)
          best_food_cluster=max(successor_food_clusters,
                                key=lambda item:item[1])[0]
          myPos = successor.getAgentState(self.index).getPosition()
          distance_to_food_cluster = self.getMazeDistance(
              myPos, best_food_cluster)
          print "Distance:",myPos,"-",best_food_cluster,"=>",distance_to_food_cluster
          features['distanceToFoodCluster'] = distance_to_food_cluster
        """

        # Compute distance to the nearest food. If it's in mode safe it shouldnt look for food.
        if len(foodList) > 0 and self.food_carrying < self.MAX_FOOD_CARRYING:
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food)
                               for food in foodList])
            features['distanceToFood'] = minDistance

        # Compute distance to closest ghost
        myPos = successor.getAgentState(self.index).getPosition()
        enemies = [successor.getAgentState(i)
                   for i in self.getOpponents(successor)]
        inRange = filter(
            lambda x: not x.isPacman and x.getPosition() != None, enemies)
        if len(inRange) > 0:
            positions = [agent.getPosition() for agent in inRange]
            closest = min(
                positions, key=lambda x: self.getMazeDistance(myPos, x))
            closestDist = self.getMazeDistance(myPos, closest)
            if closestDist <= 5:
                features['distanceToGhost'] = closestDist
                self.scared += 1
            else:
                self.scared = 0

        # featured in danger

        features['inDanger'] = 1 if self.scared else 0
        # Compute if is pacman
        features['isPacman'] = 1 if successor.getAgentState(
            self.index).isPacman else 0

        return features

    def getWeights(self, gameState, action):
        """
        Get weights for the features used in the evaluation.
        """

        successor = self.getSuccessor(gameState, action)
        myPos = successor.getAgentState(self.index).getPosition()
        enemies = [successor.getAgentState(i)
                   for i in self.getOpponents(successor)]
        inRange = filter(
            lambda x: not x.isPacman and x.getPosition() != None, enemies)
        if len(inRange) > 0:
            positions = [agent.getPosition() for agent in inRange]
            closestPos = min(
                positions, key=lambda x: self.getMazeDistance(myPos, x))
            closestDist = self.getMazeDistance(myPos, closestPos)
            closest_enemies = filter(
                lambda x: x[0] == closestPos, zip(positions, inRange))
            for agent in closest_enemies:
                if agent[1].scaredTimer > 0:
                    return {'successorScore': 200, 'distanceToFood': -5, 'distanceToGhost': 0, 'isPacman': 0}

        # return {'successorScore': 200, 'distanceToFood': -5, 'distanceToGhost': 2, 'isPacman': 0}
        return {'successorScore': 200, 'distanceToFood': -5, 'distanceToGhost': 10, 'isPacman': 0, 'inDanger': 5}
        # return {'successorScore': 200,'distanceToFood': -5, 'distanceToFoodCluster':-2, 'distanceToGhost': 2, 'isPacman': 0}

    def randomSimulation(self, depth, gameState):
        """
        Random simulate some actions for the agent. The actions other agents can take
        are ignored, or, in other words, we consider their actions is always STOP.
        The final state from the simulation is evaluated.
        """
        new_state = gameState.deepCopy()
        while depth > 0:
            # Get valid actions
            actions = new_state.getLegalActions(self.index)
            # The agent should not stop in the simulation
            actions.remove(Directions.STOP)
            current_direction = new_state.getAgentState(
                self.index).configuration.direction
            # get the reversed direction for the current direction
            reversed_direction = Directions.REVERSE[current_direction]
            # if is not the only move possible then discard it
            if reversed_direction in actions and len(actions) > 1:
                actions.remove(reversed_direction)
            # Randomly chooses a valid action
            a = random.choice(actions)
            # Compute new state and update depth
            new_state = new_state.generateSuccessor(self.index, a)
            depth -= 1
        # Evaluate the final simulation state
        return self.evaluate(new_state, Directions.STOP)

    def __init__(self, index):
        CaptureAgent.__init__(self, index)
        self.MAX_ITERATIONS = 15
        self.MAX_EXPLORATION = 20
        self.MAX_FOOD_CARRYING = 3
        self.scared = 0
        # Variables used to verify if the agent is locked

    # (15s max).
    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        # calculate all the distances
        self.distancer.getMazeDistances()

    #  (1s max).
    def chooseAction(self, gameState):
        # You can profile your evaluation time by uncommenting these lines
        start = time.time()

        # Get valid actions. Staying put is almost never a good choice, so
        # the agent will ignore this action.
        all_actions = gameState.getLegalActions(self.index)
        all_actions.remove(Directions.STOP)

        self.food_carrying = gameState.getAgentState(self.index).numCarrying

        q_a_s = {}
        for a in all_actions:
            new_state = gameState.generateSuccessor(self.index, a)
            sim_value = 0
            for iteration in range(self.MAX_ITERATIONS):
                sim_value += self.randomSimulation(
                    self.MAX_EXPLORATION, new_state)
            q_a_s[a] = sim_value

        # print "ACTIONS:",self.index,q_a_s
        next_play = max(q_a_s, key=q_a_s.get)

        # print 'eval time for offensive agent %d: %.4f, Action:%s' % (self.index, time.time() - start,next_play)
        return next_play


def writeToFile(self, gameState):
    food = self.getFood(gameState).asList()
    opponentFood = self.getFoodYouAreDefending(gameState).asList()
    score = self.getScore(gameState)
    capsules = self.getCapsules(gameState)
    capsulesOpponent = self.getCapsulesYouAreDefending(gameState)
    # print "Food:", len(food), "Defending Food :",len(opponentFood),"Score :",score


# experimantal test with UCT
class UCTBasedAgent(CaptureAgent):
    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        # print "Features",features
        # print "Weights",weights
        # print "Eval",features * weights
        return features * weights

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)
        return features

    def getWeights(self, gameState, action):
        return {'successorScore': 1.0}


class OffensiveUCT(UCTBasedAgent):

    def getFeatures(self, gameState, action):
        """
        Get features used for state evaluation.
        """
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        # Compute score from successor state
        features['successorScore'] = self.getScore(gameState)

        if self.food_carrying >= self.MAX_FOOD_CARRYING or len(self.getFood(gameState).asList()) <= 2:
            features['successorScore'] = self.getScore(gameState)
        else:
            features['successorScore'] = -len(self.getFood(gameState).asList())

        foodList = self.getFood(successor).asList()

        # Compute distance to the nearest food. If it's in mode safe it shouldnt look for food.
        if len(foodList) > 0 and self.food_carrying < self.MAX_FOOD_CARRYING:
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food)
                               for food in foodList])
            features['distanceToFood'] = minDistance

        # Compute distance to closest ghost
        myPos = successor.getAgentState(self.index).getPosition()
        enemies = [successor.getAgentState(i)
                   for i in self.getOpponents(successor)]
        inRange = filter(
            lambda x: not x.isPacman and x.getPosition() != None, enemies)
        if len(inRange) > 0:
            positions = [agent.getPosition() for agent in inRange]
            closest = min(
                positions, key=lambda x: self.getMazeDistance(myPos, x))
            closestDist = self.getMazeDistance(myPos, closest)
            if closestDist <= 5:
                features['distanceToGhost'] = closestDist

        # Compute if is pacman
        features['isPacman'] = 1 if successor.getAgentState(
            self.index).isPacman else 0

        return features

    def getWeights(self, gameState, action):
        """
        Get weights for the features used in the evaluation.
        """

        successor = self.getSuccessor(gameState, action)
        myPos = successor.getAgentState(self.index).getPosition()
        enemies = [successor.getAgentState(i)
                   for i in self.getOpponents(successor)]
        inRange = filter(
            lambda x: not x.isPacman and x.getPosition() != None, enemies)
        if len(inRange) > 0:
            positions = [agent.getPosition() for agent in inRange]
            closestPos = min(
                positions, key=lambda x: self.getMazeDistance(myPos, x))
            closestDist = self.getMazeDistance(myPos, closestPos)
            closest_enemies = filter(
                lambda x: x[0] == closestPos, zip(positions, inRange))
            for agent in closest_enemies:
                if agent[1].scaredTimer > 0:
                    return {'successorScore': 200, 'distanceToFood': -5, 'distanceToGhost': 0, 'isPacman': 0}

        return {'successorScore': 200, 'distanceToFood': -5, 'distanceToGhost': 2, 'isPacman': 0}

    def randomSimulation(self, depth, gameState):
        """
        Random simulate some actions for the agent. The actions other agents can take
        are ignored, or, in other words, we consider their actions is always STOP.
        The final state from the simulation is evaluated.
        """
        new_state = gameState.deepCopy()
        while depth > 0:
            # Get valid actions
            actions = new_state.getLegalActions(self.index)
            # The agent should not stop in the simulation
            actions.remove(Directions.STOP)
            current_direction = new_state.getAgentState(
                self.index).configuration.direction
            # get the reversed direction for the current direction
            reversed_direction = Directions.REVERSE[current_direction]
            # if is not the only move possible then discard it
            if reversed_direction in actions and len(actions) > 1:
                actions.remove(reversed_direction)
            # Randomly chooses a valid action
            a = random.choice(actions)
            # Compute new state and update depth
            new_state = new_state.generateSuccessor(self.index, a)
            depth -= 1

        # Evaluate the final simulation state
        return self.evaluate(new_state, Directions.STOP)

    def __init__(self, index):
        CaptureAgent.__init__(self, index)
        self.MAX_ITERATIONS = 15
        self.MAX_EXPLORATION = 20
        self.MAX_FOOD_CARRYING = 3

        self.cp = 0.5
        # Variables used to verify if the agent is locked

    # (15s max).
    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        # calculate all the distances
        self.distancer.getMazeDistances()

    #  (1s max).
    def chooseAction(self, gameState):
        # You can profile your evaluation time by uncommenting these lines
        start = time.time()

        # Get valid actions. Staying put is almost never a good choice, so
        # the agent will ignore this action.
        all_actions = gameState.getLegalActions(self.index)
        all_actions.remove(Directions.STOP)

        self.food_carrying = gameState.getAgentState(self.index).numCarrying
        ucb = {}
        q_a_s = {}
        for a in all_actions:
            sim_value = 0
            for iteration in range(self.MAX_ITERATIONS):
                new_state = gameState.generateSuccessor(self.index, a)
                # imulate until reach computational limit (set to MAX EXPLORATION). Backpropagate and store in sim_value the reward
                sim_value += self.randomSimulation(
                    self.MAX_EXPLORATION, new_state)

            # as we visit a new state in every iteration the number of visits = number of iterations
            q_a_s[a] = sim_value/self.MAX_ITERATIONS  # avg reward

            # n_s = number of times the state has been visited, for every action we iterated MAX_ITERATION times
            n_s = self.MAX_ITERATIONS*len(all_actions)

            # n_s_a=number of tiems I chose action a at this statte = number of iterations
            n_s_a = self.MAX_EXPLORATION
            ucb[a] = q_a_s[a]+2*self.cp*math.sqrt(2*math.log(n_s)/n_s_a)

        # print "ACTIONS:",self.index,q_a_s
        next_play = max(ucb, key=ucb.get)

        #print 'eval time for offensive agent %d: %.4f' % (self.index, time.time() - start)
        return next_play


class DefensiveUCT(UCTBasedAgent):

    def getFeatures(self, gameState, action):
        """
        Get features used for state evaluation.
        """
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        # myState =
        myPos = successor.getAgentState(self.index).getPosition()

        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i)
                   for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition()
                    != None]
        features['numInvaders'] = len(invaders)

        if len(invaders) > 0:
            dists = [self.getMazeDistance(
                myPos, a.getPosition()) for a in invaders]
            closestDist = min(dists)

            if closestDist <= 5:
                features['invaderDistance'] = closestDist

        # Compute if is pacman
        features['onDefense'] = 1 if successor.getAgentState(
            self.index).isPacman else 0

        # distance to food to secure
        # foodList=self.getFoodYouAreDefending(gameState).asList()

        """
        # Compute distance to the nearest food. If it's in mode safe it shouldnt look for food.
        if len(foodList) > 0 and len(invaders)<1:
            # myPos = successor.getAgentState(self.index).getPosition()
            # minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            # features['distanceToFoodYouAreDefending'] = minDistance

            myPos = successor.getAgentState(self.index).getPosition()
            # print(self.currentPosition,"|",myPos,"->",self.farthestFoodYouAreDefending)
            dist=self.getMazeDistance(myPos,self.farthestFoodYouAreDefending)
            features['distanceToFoodYouAreDefending'] =dist
        """

        if len(self.foodToSecure) > 0:
            minDistance = min([self.getMazeDistance(myPos, food[0])
                               for food in self.foodToSecure])
            features['distanceToFoodToSecure'] = minDistance

        # print ('Feat:',features)
        return features

    def getWeights(self, gameState, action):
        """
        Get weights for the features used in the evaluation.
        """

        # return {'distanceToFoodYouAreDefending':-200,'numInvaders': -500, 'onDefense': 100, 'invaderDistance': -1000}
        return {'distanceToFoodToSecure': -5, 'numInvaders': -3, 'onDefense': 0, 'invaderDistance': 200}

    def randomSimulation(self, depth, gameState):
        """
        Random simulate some actions for the agent. The actions other agents can take
        are ignored, or, in other words, we consider their actions is always STOP.
        The final state from the simulation is evaluated.
        """
        new_state = gameState.deepCopy()
        while depth > 0:
            # Get valid actions
            actions = new_state.getLegalActions(self.index)
            # The agent should not stop in the simulation
            actions.remove(Directions.STOP)

            current_direction = new_state.getAgentState(
                self.index).configuration.direction

            # get the reversed direction for the current direction
            reversed_direction = Directions.REVERSE[current_direction]
            # if is not the only move possible then discard it
            if reversed_direction in actions:
                if len(actions) > 1:
                    actions.remove(reversed_direction)
                else:
                    break
            # Randomly chooses a valid action
            a = random.choice(actions)

            # Compute new state and update depth
            new_state = new_state.generateSuccessor(self.index, a)
            depth -= 1

        # Evaluate the final simulation state
        self.debugDraw(new_state.getAgentState(
            self.index).getPosition(), (100, 100, 200), clear=True)
        return self.evaluate(new_state, Directions.STOP)

    def __init__(self, index):
        CaptureAgent.__init__(self, index)
        self.MAX_ITERATIONS = 15
        self.MAX_EXPLORATION = 20
        self.MAX_FOOD_CARRYING = 3
        self.farthestFoodYouAreDefending = (0, 0)
        self.foodToSecure = (0, 0)
        self.currentPosition = (0, 0)
        self.cp = 0.5
        self.initialFood = 0
        # Variables used to verify if the agent is locked

    # (15s max).
    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        # calculate all the distances
        self.distancer.getMazeDistances()

    #  (1s max).
    def chooseAction(self, gameState):
        # You can profile your evaluation time by uncommenting these lines
        start = time.time()

        # Get valid actions. Staying put is almost never a good choice, so
        # the agent will ignore this action.
        all_actions = gameState.getLegalActions(self.index)
        all_actions.remove(Directions.STOP)

        """
        # get the farest food dot
        foodList=self.getFoodYouAreDefending(gameState).asList()
        if self.initialFood==0:
            self.initialFood=len(self.getFoodYouAreDefending(gameState).asList())
        # Compute distance to the possible nearest food of the oponent

        if len(foodList) > 0:
            self.currentPosition = gameState.getAgentState(
                self.index).getPosition()
            dist={}
            for food in foodList:
                dist[food]=self.getMazeDistance(self.currentPosition, food)

            self.farthestFoodYouAreDefending = max(dist, key=dist.get)
        """
        self.foodToSecure = kmeans(self.getFoodYouAreDefending(gameState))

        self.food_carrying = gameState.getAgentState(self.index).numCarrying
        ucb = {}
        q_a_s = {}
        for a in all_actions:
            sim_value = 0
            for iteration in range(self.MAX_ITERATIONS):
                new_state = gameState.generateSuccessor(self.index, a)
                # imulate until reach computational limit (set to MAX EXPLORATION). Backpropagate and store in sim_value the reward
                sim_value += self.randomSimulation(
                    self.MAX_EXPLORATION, new_state)

            # as we visit a new state in every iteration the number of visits = number of iterations
            q_a_s[a] = sim_value/self.MAX_ITERATIONS  # avg reward

            # n_s = number of times the state has been visited, for every action we iterated MAX_ITERATION times
            n_s = self.MAX_ITERATIONS*len(all_actions)

            # n_s_a=number of tiems I chose action a at this statte = number of iterations
            n_s_a = self.MAX_EXPLORATION
            ucb[a] = q_a_s[a]+2*self.cp*math.sqrt(2*math.log(n_s)/n_s_a)

        next_play = max(ucb, key=ucb.get)
        # self.debugDraw(gameState.getAgentState(self.index).getPosition(), (100,100,200), clear=True)
        # print 'eval time for offensive agent %d: %.4f, Action:%s' % (self.index, time.time() - start,next_play)
        return next_play


class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that chooses score-maximizing actions
    """

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        start = time.time()
        actions = gameState.getLegalActions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(gameState, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        foodLeft = len(self.getFood(gameState).asList())

        if foodLeft <= 2:
            bestDist = 9999
            for action in actions:
                successor = self.getSuccessor(gameState, action)
                pos2 = successor.getAgentPosition(self.index)
                dist = self.getMazeDistance(self.start, pos2)
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist
            return bestAction

        #print 'eval time for defensive agent %d: %.4f' % (self.index, time.time() - start)
        return random.choice(bestActions)

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def getFeatures(self, gameState, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)
        return features

    def getWeights(self, gameState, action):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        return {'successorScore': 1.0}


##================BASELINE OFFENSIVE AGENT===================##
class OffensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that seeks food. This is an agent
    we give you to get an idea of what an offensive agent might look like,
    but it is by no means the best or only way to build an offensive agent.
    """

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        foodList = self.getFood(successor).asList()
        features['successorScore'] = -len(foodList)  # self.getScore(successor)

        # Compute distance to the nearest food

        if len(foodList) > 0:  # This should always be True,  but better safe than sorry
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food)
                               for food in foodList])
            features['distanceToFood'] = minDistance
        return features

    def getWeights(self, gameState, action):
        return {'successorScore': 100, 'distanceToFood': -1}


##================DEFENSIVE AGENT===================##
class GTDefensiveReflexAgent(TestReflexCaptureAgent):

    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """
    scare=0
    target_position=None
    visited=[]
    fading=7

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()
        if self.target_position==myPos:
            self.target_position=None
        ##Update self.scare
        if self.scare==0:
            self.scare=CapsuleMonitor(self, gameState,self.scare)
        elif self.scare==1:
            self.scare=CapsuleMonitor(self, gameState,self.scare)
        missingfoodinf=getMissingFood(self, gameState)

        dist_miss=0
        if len(missingfoodinf)>0:
            #Weight should be modified

            #Try to use distance to measure what action should be taken
            for pos,i in missingfoodinf:
                dist_miss+=self.getMazeDistance(pos,myPos)
        #print(missingfoodinf, action, dist_miss)
        features['MissingFood']=dist_miss


        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if myState.isPacman: features['onDefense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        alerting=[a for a in enemies if a.isPacman]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            self.target_position=None
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)
        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        if self.target_position==None and len(alerting)==0 and len(invaders)==0:
            #kmeans_positions=kmeans(self.getFoodYouAreDefending(successor))
            key_pos=keyPositions(self, gameState)
            self.target_position=key_pos[int(random.uniform(0,len(key_pos)))]

        if self.target_position!=None:
            features['targetPosition']=self.getMazeDistance(self.target_position,myPos)

        return features

    def getWeights(self, gameState, action):
        ##Priority Different
        return {'invaderDistance': -10000, 'numInvaders': -10000, 'MissingFood':-10000,'targetPosition':-100,
                'onDefense': 10000, 'stop': -10, 'reverse': -2}

def kmeans(myFood, parameter=6):
    """    
    myFood is grid variable defined in capture
       parameter is used to determine how many foods needed for a center. 
       amount of food / parameter = round down to k
       e.g  20 foods with parameter=6 gives 3 centers(round down to 3)
            20 foods with parameter=5 gives 4 centers
    """
    width=myFood.width
    height=myFood.height
    foodlist=[(i,j) for i in range(width) for j in range(height) if myFood[i][j]==True]
    k=max(1,len(foodlist)/parameter)

    if len(foodlist)>0:
        centers_=random.sample(foodlist,k)
        centers=[(i,1) for i in centers_]
        flag=0
        while(1 or flag>20):
            flag+=1
            new_clusters=[[i[0]] for i in centers]
            new_centers=[]

            for i in foodlist:
                distance=distanceCalculator.manhattanDistance(i,centers[0][0])
                index=0
                for j in range(1,len(centers)):
                    dis=distanceCalculator.manhattanDistance(i,centers[j][0])
                    if dis<distance:
                        distance=dis
                        index=j
                new_clusters[index].append(i)

            for i in range(len(new_clusters)):
                x_leng=0
                y_leng=0
                for j in range(len(new_clusters[i])):
                    x_leng+=new_clusters[i][j][0]
                    y_leng+=new_clusters[i][j][1]
                new_center=(x_leng/len(new_clusters[i]),y_leng/len(new_clusters[i]))
                dis_close = 99999
                close_food=new_clusters[i][0]
                for j in range(len(new_clusters[i])):
                    dis2=distanceCalculator.manhattanDistance(new_clusters[i][j],new_center)
                    if dis2<=dis_close:
                        dis_close=dis2
                        close_food=new_clusters[i][j]

                new_centers.append((close_food,len(new_clusters[i])))
            if (new_centers==centers):
                break;
            centers=new_centers
    return new_centers

def getMissingFood(gmagent, gameState, steps=3):
    """
    This function gives the information of missing food within previous n(default=3) steps

    This function takes gameState as input and return a list [((1,3),1), ((1,4),2), (1,5),3)]
    this means the closest food to the food was eaten in the recent one step is at position (1,3),
    and the closest food to the food that is eaten in the previous 2 step is at (1,4),
    and the closest food to the food that is eaten in the previous 3 step is at (1,5)
    that is to say the opponents pacman may be move from (1,2)->(1,3)->(1,4) accordingly.

    """

    itera = min((len(gmagent.observationHistory)-1), steps)
    ret_list = []
    for x in range(1, itera+1):
        index = -x
        preind = index-1
        curfoodlist = gmagent.getFoodYouAreDefending(
            gmagent.observationHistory[index]).asList()
        prefoodlist = gmagent.getFoodYouAreDefending(
            gmagent.observationHistory[preind]).asList()
        # print(curfoodlist)
        # print(prefoodlist)
        missingfoods = [i for i in prefoodlist if i not in curfoodlist]
        if len(missingfoods) != 0:
            missingfoods = missingfoods[0]
            dist = 9999999
            food_pos = 0
            for food in prefoodlist:
                if food != missingfoods:
                    cur_dist = gmagent.getMazeDistance(missingfoods, food)
                    if cur_dist < dist:
                        dist = cur_dist
                        food_pos = food
            ret_list.append((food_pos, x))

    return ret_list


def CapsuleMonitor(gmagent, gameState, scare, last=40):
    #print("In CapsuleMonitor")
    if scare == 0:
        index = -1
        preind = index-1
        if len(gmagent.observationHistory) > 2:
            curCaplist = gmagent.getCapsulesYouAreDefending(
                gmagent.observationHistory[index])
            preCaplist = gmagent.getCapsulesYouAreDefending(
                gmagent.observationHistory[preind])
            # print(curCaplist,preCaplist)
            if(len(preCaplist)-len(curCaplist) == 1):
                return 1
    if scare == 1 and len(gmagent.observationHistory) > 2:
        if gameState.getAgentPosition(gmagent.index) == gmagent.observationHistory[0].getAgentPosition(gmagent.index):
            return 0
    return scare


def keyPositions(gmagent, gameState):
    #curfood=gmagent.getFoodYouAreDefending(gmagent.observationHistory[0]).asList()
    half_position=(int(gameState.data.layout.width/2-gmagent.red),int(gameState.data.layout.height/2))
    while(gameState.hasWall(half_position[0],half_position[1])):
        half_position=(half_position[0],half_position[1]-1)
        #closestPos = min(curfood, key = lambda x: gmagent.getMazeDistance(half_position, x))

    FirstquaterPosition=(int((gameState.data.layout.width/2)-6*(gmagent.red-0.5)),int(gameState.data.layout.height/4))

    while(gameState.hasWall(FirstquaterPosition[0],FirstquaterPosition[1])):
        FirstquaterPosition=(FirstquaterPosition[0],FirstquaterPosition[1]-1)

    ThirdquaterPosition=(int((gameState.data.layout.width/2)-6*(gmagent.red-0.5)),int(gameState.data.layout.height*3/4))
    while(gameState.hasWall(ThirdquaterPosition[0],ThirdquaterPosition[1])):
        ThirdquaterPosition=(ThirdquaterPosition[0],ThirdquaterPosition[1]-1)
    return [half_position, FirstquaterPosition, ThirdquaterPosition]



class ApproximateQAgent(CaptureAgent):

    def __init__(self, index, epsilon=0.05, alpha=0.2, gamma=0.8, **args):
        CaptureAgent.__init__(self, index)
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.discount = float(gamma)
        self.qValues = util.Counter()
        self.lastState = None
        self.lastAction = None
        self.start = None
        self.target_position = None
        self.inLoopCount = util.Counter()
        self.max_score = 0.0

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        self.target_position = self.start
        CaptureAgent.registerInitialState(self, gameState)

        # get middle
        self.walls = gameState.getWalls()
        if(self.red):
            offset = 2
        else:
            offset = -2
        midPosition=[(self.walls.width/2 - offset, i) for i in range(1,self.walls .height-1)]
        entrances = []
        for i in midPosition:
            if not gameState.hasWall(i[0], i[1]) and i != self.start:
                entrances.append(i)
        distances = util.Counter()
        for entrance in entrances:
            dist = 0
            for food in self.getFoodYouAreDefending(gameState).asList():
                dist = dist + self.getMazeDistance(food, entrance)
            distances[entrance] = dist
        # self.entrances = distances
        self.entrances = entrances
        self.minDistantEntrance = min(distances, key=distances.get)
        self.gridSize = self.walls .width * self.walls .height
        self.initialDefendingFoodCount = len(self.getFoodYouAreDefending(gameState).asList())
        self.opponentScore = 0
        self.max_score = max(len(self.getFood(gameState).asList())-2, 1)

    def computeActionFromQValues(self, gameState):
        actions = gameState.getLegalActions(self.index)
        values = [self.getQValue(gameState, a) for a in actions]
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        return random.choice(bestActions)

    def doAction(self, state, action):
        self.lastState = state
        self.lastAction = action

    def chooseAction(self, gameState):
        # start = time.time()
        actions = gameState.getLegalActions(self.index)
        action = None
        if util.flipCoin(self.epsilon):
            action = random.choice(actions)
        else:
            action = self.computeActionFromQValues(gameState)

        self.doAction(gameState, action)  # from Q learning agent
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)
        return action

    def observeTransition(self, state, action, nextState, deltaReward):
        self.update(state, action, nextState, deltaReward)

    def getSuccessor(self, gameState, action):
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def getQValue(self, state, action):
        weights = self.getWeights()
        features = self.getFeatures(state, action)

        return weights * features

    def getWeights(self):
        return self.weights

    def update(self, state, action, nextState, reward):
        actions = nextState.getLegalActions(self.index)
        values = [self.getQValue(nextState, a) for a in actions]
        maxValue = max(values)
        weights = self.getWeights()
        features = self.getFeatures(state, action)
        difference = (reward + self.discount * maxValue) - self.getQValue(state, action)
        for feature in features:
            self.weights[feature] = weights[feature] + self.alpha * difference * features[feature]


class DefensiveQAgent(ApproximateQAgent):

    def __init__(self, index, **args):
        ApproximateQAgent.__init__(self, index, **args)
        self.filename = "teams/Poison/kazi_defensive.agent.weights"
        self.weights = util.Counter()

        if os.path.exists(self.filename):
            with open(self.filename, "r") as f:
                self.weights = pickle.load(f)
        else:
            self.weights = util.Counter({
                'bias': -4.952049116175205,
                'missingFoodDistance': -8.12789923148146,
                'distanceToEntrance': -6.622842281335308,
                'scaredState': 1.1355706099852891,
                'isPacman': -0.22433502168640782,
                'numOfInvaders': 1.0752513842357354,
                'invaderDistance': -18.058777676262988
            })

    def getFeatures(self, state, action):
        features = util.Counter()
        myPosition = state.getAgentState(self.index).getPosition()
        successor = self.getSuccessor(state, action)
        newState = successor.getAgentState(self.index)
        newPos = newState.getPosition()
        self.inLoopCount[newPos] = self.inLoopCount[newPos] + 1
        invaders = self.getInvaders(state)
        missingFoods = self.getMissingFoods(state)
        features["bias"] = 1.0
        features['numOfInvaders'] = len(invaders)
        if self.target_position == newPos:
            if self.getScore(successor) > 2 or state.getAgentState(self.index).numCarrying >= self.carryLimit:
                entrances = self.entrances
                distances = util.Counter()
                for entrance in entrances:
                    dist = 0
                    for food in self.getFoodYouAreDefending(state).asList():
                        dist = dist + self.getMazeDistance(food, entrance)
                    distances[entrance] = dist
                keyPos = min(distances, key=distances.get)
                self.target_position = keyPos
            else:
                foods = self.getFood(successor).asList()
                closest = min(foods, key=lambda x: self.getMazeDistance(newPos, x))
                self.target_position = closest
        features['invaderDistance'] = 0.0
        distanceToInvaders = [0]
        if len(invaders) > 0:
            distanceToInvaders = [self.getMazeDistance(newPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(distanceToInvaders) * 1.0 / self.gridSize
            features["isPacman"] = 0.0
            if newState.isPacman:
                features["isPacman"] = -1.0 * min(distanceToInvaders) * 1.0 / self.gridSize
        features['scaredState'] = 0.0
        if newState.scaredTimer > 0:
            features['scaredState'] = (min(distanceToInvaders) - newState.scaredTimer) * 1.0 / self.gridSize

        dist_miss = 0.0
        if len(missingFoods) > 0:
            self.target_position = missingFoods[0][0]
            for pos, i in missingFoods:
                dist_miss += self.getMazeDistance(pos, newPos)
        features['missingFoodDistance'] = dist_miss * 1.0 / self.gridSize

        minDistEntrance = self.getMazeDistance(newPos, self.target_position)
        features['distanceToEntrance'] = minDistEntrance * 1.0 / self.gridSize

        return features

    def observationFunction(self, state):
        if self.lastState:
            reward = self.getRewards(state, self.lastState)
            self.observeTransition(self.lastState, self.lastAction, state, reward)

        return CaptureAgent.observationFunction(self, state)

    def getRewards(self, state, lastState):
        reward = self.getRecoveredFoodCount(state, lastState)
        reward -= len(self.getInvaders(state)) - len(self.getInvaders(lastState))
        reward -= 1
        distancePosition = self.getMazeDistance(state.getAgentState(self.index).getPosition(), lastState.getAgentState(self.index).getPosition())
        if distancePosition > 1:
            reward -= distancePosition * 1.0/self.gridSize
        return reward

    def getInvaders(self, state):
        enemies = [state.getAgentState(i) for i in self.getOpponents(state)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition()]

        return invaders

    def getRecoveredFoodCount(self, state, lastState):
        return len(self.getFoodYouAreDefending(state).asList()) - len(self.getFoodYouAreDefending(lastState).asList())

    def getMissingFoods(self, gameState, steps=6):
        itera = min((len(self.observationHistory) - 1), steps)
        ret_list = []
        for x in range(1, itera + 1):
            index = -x
            preind = index - 1
            curfoodlist = self.getFoodYouAreDefending(self.observationHistory[index]).asList()
            prefoodlist = self.getFoodYouAreDefending(self.observationHistory[preind]).asList()
            missingfoods = [i for i in prefoodlist if i not in curfoodlist]
            if len(missingfoods) != 0:
                missingfoods = missingfoods[0]
                dist = 9999999
                food_pos = prefoodlist[0]
                for food in prefoodlist:
                    if food != missingfoods:
                        cur_dist = self.getMazeDistance(missingfoods, food)
                        if cur_dist < dist:
                            dist = cur_dist
                            food_pos = food
                ret_list.append((food_pos, x))

        return ret_list

    def computeActionFromQValues(self, gameState):
        actions = gameState.getLegalActions(self.index)
        actions.remove('Stop')
        values = [self.getQValue(gameState, a) for a in actions]

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        best = random.choice(bestActions)
        return best