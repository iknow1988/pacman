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
from game import Actions
sys.path.append('teams/Poison/')
import pickle

random.seed(42)
#################
# Team creation #
#################


def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveQAgent', second='DefensiveQAgent'):
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
        self.max_score = 0.0
        self.entrances = []
        self.minDistantEntrance = None
        self.gridSize = None
        self.walls = []

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
        # get middle
        self.walls = gameState.getWalls()
        if self.red:
            offset = 2
        else:
            offset = -2
        midPosition = [(self.walls.width / 2 - offset, i) for i in range(1, self.walls.height - 1)]
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
        self.entrances = entrances
        self.minDistantEntrance = min(distances, key=distances.get)
        self.gridSize = self.walls.width * self.walls.height
        self.max_score = max(len(self.getFood(gameState).asList()) - 2, 1)

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
        actions = gameState.getLegalActions(self.index)
        action = None
        action = self.computeActionFromQValues(gameState)
        self.doAction(gameState, action)  # from Q learning agent
        return action

    def observationFunction(self, state):
        if self.lastState:
            reward = self.getRewards(state, self.lastState)
            self.observeTransition(self.lastState, self.lastAction, state, reward)
        return CaptureAgent.observationFunction(self, state)

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

    def observeTransition(self, state, action, nextState, deltaReward):
        self.update(state, action, nextState, deltaReward)

    def update(self, state, action, nextState, reward):
        actions = nextState.getLegalActions(self.index)
        values = [self.getQValue(nextState, a) for a in actions]
        maxValue = max(values)
        weights = self.getWeights()
        features = self.getFeatures(state, action)
        difference = (reward + self.discount * maxValue) - self.getQValue(state, action)
        for feature in features:
            self.weights[feature] = weights[feature] + self.alpha * difference * features[feature]

    def aStarSearch(self, gameState, goalPositions, startPosition=None, avoidPositions=[], returngoalPosition=False,
                    returnCost=False):
        gmagent = self
        walls = gameState.getWalls()
        width = walls.width
        height = walls.height
        walls = walls.asList()
        if startPosition == None:
            startPosition = gmagent.getCurrentObservation().getAgentPosition(gmagent.index)

        actions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
        actionVectors = [(int(Actions.directionToVector(action)[0]), int(Actions.directionToVector(action)[1])) for
                         action in actions]
        currentPosition, currentPath, currentCost = startPosition, [], 0

        queue = util.PriorityQueueWithFunction(lambda entry: entry[2] +  # Total cost so far
                                                             min(self.getMazeDistance(entry[0], endPosition) for
                                                                 endPosition in goalPositions))
        # width * height if entry[0] in avoidPositions else 0 +  # Avoid enemy locations like the plague

        # No Revisits
        visited = set([currentPosition])

        while currentPosition not in goalPositions:
            possiblePositions = [((currentPosition[0] + vector[0], currentPosition[1] + vector[1]), action) for
                                 vector, action in zip(actionVectors, actions)]
            legalPositions = [(position, action) for position, action in possiblePositions if position not in walls]
            for position, action in legalPositions:
                if position not in visited:
                    visited.add(position)
                    AvoidValue = 0
                    if position in avoidPositions:
                        AvoidValue = width * height

                    """value2=util.manhattanDistance(position, goalPositions[0])

                    for endPosition in goalPositions:
                        if util.manhattanDistance(position, endPosition) <value2:
                            value2=util.manhattanDistance(position, endPosition)"""

                    queue.push((position, currentPath + [action], currentCost + 1 + AvoidValue))
            if queue.isEmpty():  ##Just in case
                return None
            else:
                currentPosition, currentPath, currentCost = queue.pop()
        if returnCost:
            return currentPath, currentPosition, currentCost

        if returngoalPosition:
            return currentPath, currentPosition

        else:
            return currentPath
        return currentPath

    def CheckIfAgentStucking(self, gameState, referhistory=10, countingfactor=3):
        gmagent = self
        referhistory = min(referhistory, len(gmagent.observationHistory))
        curposition = gameState.getAgentPosition(gmagent.index)

        for i in range(-1, -1 - referhistory, -1):
            historyposition2 = gmagent.observationHistory[i].getAgentPosition(gmagent.index)
            if curposition == historyposition2:
                countingfactor -= 1
        return countingfactor < 0

    def FindAlternativeFood(self, gameState, returngoalPosition=True):
        gmagent = self
        myPos = gmagent.getCurrentObservation().getAgentPosition(gmagent.index)
        enemies = [gameState.getAgentState(i) for i in gmagent.getOpponents(gameState)]
        chasers = []
        for a in enemies:
            if not a.isPacman:
                if a.getPosition() != None:
                    chasers.append(a.getPosition())

        # chasers = [a.getPosition() for a in enemies if  (not a.isPacman) and a.getPosition() != None]
        walls = gameState.getWalls()

        foodList = gmagent.getFood(gameState).asList()

        for cap in gmagent.getCapsules(gameState):
            foodList.append(cap)

        height = walls.height
        width = walls.width
        walls = walls.asList()
        half_position = (int(gameState.data.layout.width / 2 - gmagent.red), int(gameState.data.layout.height / 2))
        while (gameState.hasWall(half_position[0], half_position[1])):
            half_position = (half_position[0], half_position[1] - 1)

        goalPositions = foodList

        avoidPos = []
        X = min(width / 4, 3)
        Y = min(height / 4, 3)

        for chaser in chasers:
            for posX in range(int(max(1, chaser[0] - X)), int(min(width, chaser[0] + X))):
                for posY in range(int(max(0, chaser[1] - Y)), int(min(height, chaser[1] + Y))):
                    if not gameState.hasWall(posX, posY):
                        if (abs(posX - chaser[0]) + abs(posY - chaser[1])) <= 2:
                            avoidPos.append((posX, posY))
                        if (posX, posY) in goalPositions:
                            goalPositions.remove((posX, posY))
        if len(goalPositions) == 0:
            return None, None
        ##Here return a list and the position
        currentPath, currentPosition = self.aStarSearch(gameState, goalPositions=goalPositions, startPosition=myPos,
                                                        avoidPositions=avoidPos, returngoalPosition=True)

        steps = min(5, len(currentPath))
        stackpath = []
        if steps > 0:
            for i in range(steps - 1, -1, -1):
                stackpath.append(currentPath[i])
        return stackpath, currentPosition

    def getSafeActions(self, gameState, actions):
        safeActions = []
        capsules = self.getCapsules(gameState)
        for action in actions:
            if not self.target_position in capsules and not self.RunForestCheckDeadAlley(gameState, action):
                safeActions.append(action)

        return safeActions

    def RunForestCheckDeadAlley(self, gameState, action):
        """
        Call this function when you are in urgent running ignoring foods
        RETURN: True when this direction is dangerous
        """
        gmagent = self
        walls = gameState.getWalls()
        width = walls.width
        height = walls.height
        walls = walls.asList()
        startPosition = gmagent.getCurrentObservation().getAgentPosition(gmagent.index)
        avoidPos = [startPosition]

        half_position = (int(gameState.data.layout.width / 2 - gmagent.red), int(gameState.data.layout.height / 2))
        while (gameState.hasWall(half_position[0], half_position[1])):
            half_position = (half_position[0], half_position[1] - 1)

        goalPositions = [(half_position[0], height_position) for height_position in range(3, height - 1) if
                         not gameState.hasWall(half_position[0], height_position)]

        successor = gmagent.getSuccessor(gameState, action)

        myState = successor.getAgentState(gmagent.index)
        successorPos = myState.getPosition()
        Path, Position, Cost = self.aStarSearch(gameState, goalPositions, startPosition=successorPos,
                                                avoidPositions=avoidPos, returngoalPosition=False, returnCost=True)
        if Cost > width * height:
            return True
        # width * height
        return False

    def isOpponentScared(self, state):
        scared = False
        enemies = [state.getAgentState(i) for i in self.getOpponents(state)]
        for a in enemies:
            if not a.isPacman:
                if a.scaredTimer > 3:
                    scared = True
                    break

        return scared

    def getInvaders(self, state):
        enemies = [state.getAgentState(i) for i in self.getOpponents(state)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition()]

        return invaders

    def getGhosts(self, state):
        enemies = [state.getAgentState(i) for i in self.getOpponents(state)]
        ghosts = [a for a in enemies if not a.isPacman and a.getPosition()]

        return ghosts

    def LoopBreakerMoniter(self, gameState, referhistory=24):
        gmagent = self
        if len(gmagent.observationHistory) <= referhistory: return False

        myPos = gmagent.getCurrentObservation().getAgentPosition(gmagent.index)
        enemies = [gameState.getAgentState(i) for i in gmagent.getOpponents(gameState)]
        chaser = [a.getPosition() for a in enemies if (not a.isPacman) and a.getPosition() != None]

        if len(chaser) > 0:
            referDistance = gmagent.getMazeDistance(myPos, chaser[0])
        else:
            return False
        for i in range(-referhistory, -1):
            myPreState = gmagent.observationHistory[i]
            myPrePos = myPreState.getAgentPosition(gmagent.index)
            Preenemies = [myPreState.getAgentState(i) for i in gmagent.getOpponents(myPreState)]
            Prechaser = [a.getPosition() for a in Preenemies if (not a.isPacman) and a.getPosition() != None]
            if len(Prechaser) == 0:
                return False
            PreDist = gmagent.getMazeDistance(myPrePos, Prechaser[0])
            if PreDist != referDistance:
                return False

        return True


class OffensiveQAgent(ApproximateQAgent):

    def __init__(self, index, **args):
        ApproximateQAgent.__init__(self, index, **args)
        self.filename = "final.offensive.agent.weights"
        self.weights = util.Counter({
            'distanceToGhost': -1.1817616960168724,
            'bias': -4.807251464599814,
            'numOfGhosts': -0.3594832835217913,
            'targetPosition': -22.31644664126268,
            'distanceToInvader': -0.523948237607476
        })
        self.target_position = None
        self.carryLimit = self.max_score
        self.alternativePath = []

    def registerInitialState(self, gameState):
        ApproximateQAgent.registerInitialState(self, gameState)
        self.target_position = min(self.entrances,
                                   key=lambda x: self.getMazeDistance(gameState.getAgentState(self.index).getPosition(),
                                                                      x))
        if len(self.getFood(gameState).asList()) > 0:
            self.target_position = max(self.getFood(gameState).asList(),
                                       key=lambda x: self.getMazeDistance(
                                           gameState.getAgentState(self.index).getPosition(),
                                           x))
        self.carryLimit = self.max_score

    def getFeatures(self, state, action):
        myCurrentState = state.getAgentState(self.index)
        myCurrentPosition = myCurrentState.getPosition()
        successor = self.getSuccessor(state, action)
        foodList = self.getFood(state).asList()
        myNextState = successor.getAgentState(self.index)
        myNextPosition = myNextState.getPosition()
        ghosts = self.getGhosts(state)
        invaders = self.getInvaders(state)
        features = util.Counter()
        minDistanceToGhost = 0.0
        minDistToGhost = 0.0
        minDistanceToInvader = 0.0
        capsules = self.getCapsules(state)
        minDistantCapsule = 0
        if len(capsules) > 0:
            minDistantCapsule = max(capsules, key=lambda x: self.getMazeDistance(myNextPosition, x))

        if len(ghosts) > 0:
            distancesToGhosts = [self.getMazeDistance(myNextPosition, a.getPosition()) for a in ghosts]
            minDistToGhost = min(distancesToGhosts)
            if not self.isOpponentScared(successor) and minDistToGhost <= 1 and myNextState.isPacman:
                minDistanceToGhost = minDistToGhost * 1.0
            if minDistToGhost > 5 or self.isOpponentScared(successor):
                self.carryLimit = self.max_score
            else:
                self.carryLimit = 10

        if len(invaders) > 0:
            distancesToInvaders = [self.getMazeDistance(myNextPosition, a.getPosition()) for a in invaders]
            if myNextState.isPacman and min(distancesToInvaders) <= 1:
                minDistanceToInvader = min(distancesToInvaders) * 1.0

        # eaten a food, giving another food to eat
        if myCurrentPosition == self.target_position and len(foodList) > 2:
            self.target_position = min(foodList, key=lambda x: self.getMazeDistance(myNextPosition, x))

        # eaten everything go back home
        if len(foodList) <= 2:
            self.target_position = min(self.entrances, key=lambda x: self.getMazeDistance(myNextPosition, x))

        # reached carry limit go back to home
        if not self.isOpponentScared(state) and len(ghosts) > 0 and \
                minDistToGhost <= 5 and myCurrentState.numCarrying >= self.carryLimit:
            self.target_position = min(self.entrances, key=lambda x: self.getMazeDistance(myNextPosition, x))

        # End of time and go back to home
        timeLeft = state.data.timeleft * 1.0 / state.getNumAgents()
        target = min(self.entrances, key=lambda x: self.getMazeDistance(myNextPosition, x))
        dist = self.getMazeDistance(myNextPosition, target)
        if timeLeft - dist <= 2.0:
            self.target_position = target

        # no need to eat capsule if not needed
        if len(capsules) > 0 and not self.isOpponentScared(state) and len(ghosts) > 0:
            minDistanceToTarget = self.getMazeDistance(myNextPosition, self.target_position)
            minDistanceToCapsule = self.getMazeDistance(myNextPosition, minDistantCapsule)
            if minDistanceToTarget > minDistanceToCapsule:
                self.target_position = minDistantCapsule

        # food eating coordination with defender
        if self.lastState:
            lastStatefoodList = self.getFood(self.lastState).asList()
            if self.target_position in lastStatefoodList and self.target_position not in foodList:
                self.target_position = min(foodList, key=lambda x: self.getMazeDistance(myNextPosition, x))

        # Loop Braking monitor
        if ((len(ghosts) != 0) or not self.isOpponentScared(state)) and self.LoopBreakerMoniter(state):
            self.target_position = min(self.entrances, key=lambda x: self.getMazeDistance(myNextPosition, x))

        features["bias"] = 1.0
        features['numOfGhosts'] = len(ghosts) * 1.0 / 2.0
        features['distanceToGhost'] = minDistanceToGhost
        features['distanceToInvader'] = minDistanceToInvader
        features['targetPosition'] = self.getMazeDistance(myNextPosition, self.target_position) * 1.0 / self.gridSize

        # self.debugDraw(self.target_position, (1, 0, 0), clear=True)

        return features

    def chooseAction(self, gameState):
        action = None
        if len(self.alternativePath) > 0:
            action = self.alternativePath.pop()
        elif not self.CheckIfAgentStucking(gameState):
            actions = gameState.getLegalActions(self.index)
            action = self.computeActionFromQValues(gameState)
        else:
            actions, goalDestination = self.FindAlternativeFood(gameState)
            if actions:
                self.alternativePath = actions
                action = self.alternativePath.pop()
                self.target_position = goalDestination
            else:
                actions = gameState.getLegalActions(self.index)
                action = random.choice(actions)

        self.doAction(gameState, action)  # from Q learning agent

        return action

    def observationFunction(self, state):
        if self.lastState:
            distancePosition = self.getMazeDistance(state.getAgentState(self.index).getPosition(),
                                                    self.lastState.getAgentState(self.index).getPosition())
            if distancePosition > 1:
                self.alternativePath = []
                if (len(self.getFood(state).asList())) > 0:
                    self.target_position = max(self.getFood(state).asList(),
                                               key=lambda x: self.getMazeDistance(
                                                   state.getAgentState(self.index).getPosition(),
                                                   x))
            if len(self.alternativePath) == 0:
                reward = self.getRewards(state, self.lastState)
                self.observeTransition(self.lastState, self.lastAction, state, reward)

        return CaptureAgent.observationFunction(self, state)

    def getRewards(self, state, lastState):
        reward = 0
        myPosition = state.getAgentState(self.index).getPosition()
        lastPosition = lastState.getAgentState(self.index).getPosition()
        targetPosition = self.target_position
        foodList = self.getFood(lastState).asList()
        capsule = self.getCapsules(lastState)
        if targetPosition != myPosition:
            reward -= 1
        else:
            if myPosition in foodList:
                reward += 1
            elif myPosition in capsule:
                reward += 2
            else:
                reward += self.getScore(state) - self.getScore(lastState)
        distanceToPreviousStateLocation = self.getMazeDistance(myPosition, lastPosition)
        if distanceToPreviousStateLocation > 1:
            reward -= distanceToPreviousStateLocation * 1.0 / self.gridSize

        return reward

    def computeActionFromQValues(self, gameState):
        actions = gameState.getLegalActions(self.index)
        actions.remove('Stop')
        ghosts = self.getGhosts(gameState)
        myCurrentState = gameState.getAgentState(self.index)
        myCurrentPosition = myCurrentState.getPosition()
        if len(ghosts) > 0:
            distancesToGhosts = [self.getMazeDistance(myCurrentPosition, a.getPosition()) for a in ghosts]
            if not self.isOpponentScared(gameState) and min(distancesToGhosts) <= 6 and myCurrentState.isPacman:
                b = self.getSafeActions(gameState, actions)
                if len(b) > 0:
                    actions = b
        values = [self.getQValue(gameState, a) for a in actions]
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        best = random.choice(bestActions)

        return best


class DefensiveQAgent(ApproximateQAgent):

    def __init__(self, index, **args):
        ApproximateQAgent.__init__(self, index, **args)
        self.target_position = None
        self.initial = False
        self.alternativePath = []
        self.initialFoodListDefending = []
        self.initialFoodListToEat = []
        self.weights = util.Counter({
            'bias': -4.7358416969746395,
            'missingFoodDistance': -4.443376568936904,
            'distanceToEntrance': -14.989341544103803,
            'scaredState': 6.2021608272609905,
            'isPacman': 0.07327131977082438,
            'numOfInvaders': 0.7315522476852717,
            'invaderDistance': -26.17572385337668
        })

    def registerInitialState(self, gameState):
        ApproximateQAgent.registerInitialState(self, gameState)
        self.target_position = self.minDistantEntrance
        self.alternativePath = []
        self.initialFoodListDefending = self.getFoodYouAreDefending(gameState).asList()
        self.initialFoodListToEat = self.getFood(gameState).asList()
        self.carryLimit = math.ceil(self.max_score * 0.20)

    def getFeatures(self, state, action):
        features = util.Counter()
        myCurrentState = state.getAgentState(self.index)
        myCurrentPosition = state.getAgentState(self.index).getPosition()
        successor = self.getSuccessor(state, action)
        newState = successor.getAgentState(self.index)
        newPosition = newState.getPosition()
        invaders = self.getInvaders(state)
        ghosts = self.getGhosts(state)
        missingFoods = self.getMissingFoods(state)
        foodListDefending = self.getFoodYouAreDefending(state).asList()
        foodListToEat = self.getFood(state).asList()
        capsules = self.getCapsules(state)
        minDistantCapsule = 0
        intercept = None
        if len(capsules) > 0:
            minDistantCapsule = min(capsules, key=lambda x: self.getMazeDistance(newPosition, x))

        if myCurrentPosition in self.entrances:
            self.initial = True
        # no need of alternative path being a ghost
        if not newState.isPacman:
            self.alternativePath = []

        # Team is losing and go offensive
        halfFood = len(self.initialFoodListDefending) * 1.0 / 2.0
        if self.initial and (self.getScore(state) != 0 and self.getScore(state) <= int(self.max_score / 8.0) and len(
                foodListDefending) > halfFood and len(
            foodListToEat) > 2 and newState.numCarrying <= self.carryLimit) \
                and len(foodListDefending) != len(self.initialFoodListDefending):
            self.target_position = min(foodListToEat, key=lambda x: self.getMazeDistance(newPosition, x))

        # goal aware target for defenses
        if self.target_position == myCurrentPosition:
            foodLeftToDefend = len(foodListDefending) * 1.0
            if foodLeftToDefend > len(self.initialFoodListDefending) * 1.0 / 2:
                distances = []
                for entrance in self.entrances:
                    distances.append([self.getMazeDistance(food, entrance) for food in foodListDefending])
                bestLocations = [a for a, v in zip(self.entrances, distances) if v == min(distances)]
                self.target_position = random.choice(bestLocations)
            elif 0 < foodLeftToDefend <= len(self.initialFoodListDefending) * 1.0 / 2:
                successor_food_clusters = self.kmeans(self.getFoodYouAreDefending(state), 2)
                best_food_cluster = max(successor_food_clusters,
                                        key=lambda item: item[1])[0]
                self.target_position = best_food_cluster
            else:
                self.target_position = min(self.entrances, key=lambda x: self.getMazeDistance(newPosition, x))

        minDistanceToInvader = 0.0
        minDistanceToGhost = 0.0
        if len(invaders) > 0:
            distanceToInvaders = [self.getMazeDistance(newPosition, a.getPosition()) for a in invaders]
            minDist = min(distanceToInvaders)
            try:
                intercept = self.InterceptOpponents(state)
            except IndexError:
                intercept = None
                print "Intercept error"
            if not intercept and not myCurrentState.isPacman and newState.scaredTimer <= 0:
                minDistanceToInvader = min(distanceToInvaders)

        if len(ghosts) > 0:
            distanceToGhosts = [self.getMazeDistance(newPosition, a.getPosition()) for a in ghosts]
            if newState.isPacman and min(distanceToGhosts) <= 1 and not self.isOpponentScared(state):
                minDistanceToGhost = -1.0 * min(distanceToGhosts)

        # Opponent ate capsule so go to offensive
        if newState.scaredTimer > 0 and len(foodListToEat) > 2 and \
                len(foodListDefending) > len(self.initialFoodListDefending) * 1.0 / 2:
            self.target_position = min(foodListToEat, key=lambda x: self.getMazeDistance(newPosition, x))

        # Eat capsule if it is near
        if myCurrentState.isPacman and len(capsules) > 0 and not self.isOpponentScared(state):
            if self.getMazeDistance(newPosition, self.target_position) > self.getMazeDistance(newPosition,
                                                                                              minDistantCapsule):
                self.target_position = minDistantCapsule

        # teammate ate the food and update to new target (coordination)
        if newState.isPacman and self.lastState:
            foodInLastState = self.target_position in self.initialFoodListToEat
            foodNotInPresentState = self.target_position not in foodListToEat
            if foodInLastState and foodNotInPresentState and len(foodListToEat) > 2:
                self.target_position = min(foodListToEat, key=lambda x: self.getMazeDistance(newPosition, x))

        # Notifying defenseive agent when opponent ate a food
        dist_miss = 0.0
        if len(missingFoods) > 0 and minDistanceToInvader > 6 and \
                not myCurrentState.isPacman and len(invaders) > 0:
            self.target_position = missingFoods[0][0]
            dist_miss = self.getMazeDistance(missingFoods[0][0], newPosition)

        # reached carry limit go back to home
        if self.initial and newState.isPacman and not self.isOpponentScared(state) \
                and myCurrentState.numCarrying > self.carryLimit:
            self.target_position = min(self.entrances, key=lambda x: self.getMazeDistance(newPosition, x))
            self.initial = False

        # End of time and go back to home
        timeLeft = state.data.timeleft * 1.0 / state.getNumAgents()
        target = min(self.entrances, key=lambda x: self.getMazeDistance(newPosition, x))
        dist = self.getMazeDistance(newPosition, target)
        if newState.isPacman and (timeLeft - dist < 2.0):
            self.target_position = target

        # # eaten everything go back home
        if newState.isPacman and len(foodListToEat) <= 2:
            self.target_position = min(self.entrances, key=lambda x: self.getMazeDistance(newPosition, x))

        if intercept:
            self.target_position = intercept
            # print "INTERCEPTING"

        if myCurrentState.isPacman and self.LoopBreakerMoniter(state):
            self.target_position = self.minDistantEntrance

        features['scaredState'] = 0.0
        if newState.scaredTimer > 0:
            features['scaredState'] = (minDistanceToInvader - newState.scaredTimer) * 1.0 / self.gridSize
        features['invaderDistance'] = minDistanceToInvader * 1.0 / self.gridSize
        features["isPacman"] = minDistanceToGhost
        features['missingFoodDistance'] = dist_miss * 1.0 / self.gridSize
        features['distanceToEntrance'] = self.getMazeDistance(newPosition, self.target_position) * 1.0 / self.gridSize
        features["bias"] = 1.0
        features['numOfInvaders'] = len(invaders) / 2

        # self.debugDraw(self.target_position, (1, 1, 1), clear=True)

        return features

    def observationFunction(self, state):
        if self.lastState:
            distancePosition = self.getMazeDistance(state.getAgentState(self.index).getPosition(),
                                                    self.lastState.getAgentState(self.index).getPosition())
            if distancePosition > 1:
                self.alternativePath = []
                if (len(self.getFood(state).asList())) > 0:
                    self.target_position = self.minDistantEntrance
            if len(self.alternativePath) == 0 and self.lastAction != 'Stop':
                reward = self.getRewards(state, self.lastState)
                self.observeTransition(self.lastState, self.lastAction, state, reward)

        return CaptureAgent.observationFunction(self, state)

    def getRewards(self, state, lastState):
        myCurrentState = state.getAgentState(self.index)
        if not myCurrentState.isPacman:
            reward = self.getRecoveredFoodCount(state, lastState)
            reward -= len(self.getInvaders(state)) - len(self.getInvaders(lastState))
            if self.target_position != state.getAgentState(self.index).getPosition():
                reward -= 1
            distancePosition = self.getMazeDistance(state.getAgentState(self.index).getPosition(),
                                                    lastState.getAgentState(self.index).getPosition())
            if distancePosition > 1:
                reward -= distancePosition * 1.0 / self.gridSize
        else:
            reward = 0
            myPosition = state.getAgentState(self.index).getPosition()
            lastPosition = lastState.getAgentState(self.index).getPosition()
            targetPosition = self.target_position
            foodList = self.getFood(lastState).asList()
            capsule = self.getCapsules(lastState)
            if targetPosition != myPosition:
                reward -= 1
            else:
                if myPosition in foodList:
                    reward += 1 / self.max_score
                elif myPosition in capsule:
                    reward += 2 / self.max_score
                else:
                    reward += self.getScore(state) - self.getScore(lastState)
                distanceToPreviousStateLocation = self.getMazeDistance(myPosition, lastPosition)
                if distanceToPreviousStateLocation > 1:
                    reward -= distanceToPreviousStateLocation * 1.0 / self.gridSize

        return reward

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
        # return 'Stop'
        actions = gameState.getLegalActions(self.index)
        actions.remove('Stop')
        ghosts = self.getGhosts(gameState)
        myCurrentState = gameState.getAgentState(self.index)
        myCurrentPosition = myCurrentState.getPosition()
        if myCurrentState.isPacman and len(ghosts) > 0:
            distancesToGhosts = [self.getMazeDistance(myCurrentPosition, a.getPosition()) for a in ghosts]
            if len(ghosts) > 0 and not self.isOpponentScared(gameState) and min(
                    distancesToGhosts) <= 4 and myCurrentState.isPacman:
                b = self.getSafeActions(gameState, actions)
                if len(b) > 0:
                    actions = b
        values = [self.getQValue(gameState, a) for a in actions]

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        best = random.choice(bestActions)

        return best

    def chooseAction(self, gameState):
        actions = gameState.getLegalActions(self.index)
        if gameState.getAgentState(self.index).isPacman and len(self.alternativePath) > 0:
            action = self.alternativePath.pop()
        elif not gameState.getAgentState(self.index).isPacman and self.CloggingOpponent(gameState):
            action = 'Stop'
        elif gameState.getAgentState(self.index).isPacman and self.CheckIfAgentStucking(gameState):
            actions, goalDestination = self.FindAlternativeFood(gameState)
            if actions:
                self.alternativePath = actions
                action = self.alternativePath.pop()
                self.target_position = goalDestination
            else:
                actions = gameState.getLegalActions(self.index)
                action = random.choice(actions)
        else:
            actions = gameState.getLegalActions(self.index)
            action = self.computeActionFromQValues(gameState)

        self.doAction(gameState, action)  # from Q learning agent

        return action

    def CloggingOpponent(self, gameState, returngoalPosition=False):
        gmagent = self
        myPos = gmagent.getCurrentObservation().getAgentPosition(gmagent.index)
        enemies = [gameState.getAgentState(i) for i in gmagent.getOpponents(gameState)]
        chaser = [a.getPosition() for a in enemies if a.isPacman and a.getPosition() != None]

        walls = gameState.getWalls()
        height = walls.height
        width = walls.width
        walls = walls.asList()
        # 22 38 792
        half_position = (int(gameState.data.layout.width / 2 - gmagent.red), int(gameState.data.layout.height / 2))
        while (gameState.hasWall(half_position[0], half_position[1])):
            half_position = (half_position[0], half_position[1] - 1)

        goalPositions = [(half_position[0], height_position) for height_position in range(3, height - 1) if
                         not gameState.hasWall(half_position[0], height_position)]
        avoidPositions = [myPos]

        startPosition = None

        if len(chaser) > 0:
            startPosition = chaser[0]

        Path, Position, Cost = self.aStarSearch(gameState, goalPositions, startPosition=startPosition,
                                                avoidPositions=avoidPositions, returngoalPosition=False,
                                                returnCost=True)

        if len(chaser) > 1:
            Path2, Position2, Cost2 = self.aStarSearch(gameState, goalPositions, startPosition=chaser[1],
                                                       avoidPositions=avoidPositions, returngoalPosition=False,
                                                       returnCost=True)
            if Cost2 > Cost:
                Cost = Cost2

        if Cost > width * height and len(chaser) > 0:
            return True

        ####NEW NEED TESTING
        # width * height
        if (len(gmagent.observationHistory) >= 10):
            index_state = -2
            myPreState = gmagent.observationHistory[index_state]
            myPrePos = myPreState.getAgentPosition(gmagent.index)

            if (myPrePos != myPos):
                return False
            while (myPrePos == myPos and (abs(index_state) < (len(gmagent.observationHistory) - 5))):
                index_state -= 1
                myPreState = gmagent.observationHistory[index_state]
                myPrePos = myPreState.getAgentPosition(gmagent.index)

            index_state += 1
            myPreState = gmagent.observationHistory[index_state]
            myPrePos = myPreState.getAgentPosition(gmagent.index)

            Preenemies = [(myPreState.getAgentState(i), i) for i in gmagent.getOpponents(myPreState)]
            Prechaser = [(a.getPosition(), i) for a, i in Preenemies if a.isPacman and a.getPosition() != None]
            goalPositions = [(half_position[0], height_position) for height_position in range(3, height - 1) if
                             not gameState.hasWall(half_position[0], height_position)]
            avoidPositions = [myPos]

            index_clog = 0
            for a, i in Prechaser:
                Path, Position, Cost = self.aStarSearch(gameState, goalPositions, startPosition=a,
                                                        avoidPositions=avoidPositions, returngoalPosition=False,
                                                        returnCost=True)
                if Cost > width * height:
                    index_clog = i
                    break
            if index_clog == 0: return False

            # print("NOW===",myPrePos==myPos, myPrePos, myPos, index_state, len(Prechaser))
            while (index_state != -2):
                myPreState = gmagent.observationHistory[index_state]
                myPrePos = myPreState.getAgentPosition(gmagent.index)
                Preenemies = [myPreState.getAgentState(i) for i in gmagent.getOpponents(myPreState) if i == index_clog]
                Prechaser = [a.getPosition() for a in Preenemies if a.isPacman and a.getPosition() != None]
                index_state += 1
                if len(Prechaser) == 0:
                    break

            index_state -= 2
            myPreState = gmagent.observationHistory[index_state]
            myPrePos = myPreState.getAgentPosition(gmagent.index)
            # print("AFTER===",myPrePos==myPos, myPrePos, myPos, index_state, len(Prechaser))
            Preenemies = [myPreState.getAgentState(i) for i in gmagent.getOpponents(myPreState)]
            Prechaser = [a.getPosition() for a in Preenemies if a.isPacman and a.getPosition() != None]
            if len(Prechaser) > 0 and gmagent.getMazeDistance(myPrePos, Prechaser[0]) >= 3: return True

        return False

    def kmeans(self, myFood, parameter=6):
        width = myFood.width
        height = myFood.height
        foodlist = [(i, j) for i in range(width) for j in range(height) if myFood[i][j] == True]
        k = max(1, len(foodlist) / parameter)
        new_centers = []
        centers = []
        if len(foodlist) == 0:
            return []
        if len(foodlist) > 0:
            centers_ = random.sample(foodlist, k)
            centers = [(i, 1) for i in centers_]
            flag = 0
            while (1 or flag > 20):
                flag += 1
                new_clusters = [[i[0]] for i in centers]

                for i in foodlist:
                    distance = distanceCalculator.manhattanDistance(i, centers[0][0])
                    index = 0
                    for j in range(1, len(centers)):
                        dis = distanceCalculator.manhattanDistance(i, centers[j][0])
                        if dis < distance:
                            distance = dis
                            index = j
                    new_clusters[index].append(i)

                for i in range(len(new_clusters)):
                    x_leng = 0
                    y_leng = 0
                    for j in range(len(new_clusters[i])):
                        x_leng += new_clusters[i][j][0]
                        y_leng += new_clusters[i][j][1]
                    new_center = (x_leng / len(new_clusters[i]), y_leng / len(new_clusters[i]))
                    dis_close = 99999
                    close_food = new_clusters[i][0]
                    for j in range(len(new_clusters[i])):
                        dis2 = distanceCalculator.manhattanDistance(new_clusters[i][j], new_center)
                        if dis2 <= dis_close:
                            dis_close = dis2
                            close_food = new_clusters[i][j]

                    new_centers.append((close_food, len(new_clusters[i])))
                if (new_centers == centers):
                    break;
                centers = new_centers
        return new_centers

    def EscapePathForOpponent(self, gameState, returngoalPosition=False):
        ##astar:gmagent, gameState, goalPositions, startPosition=None, avoidPositions=[], returngoalPosition=False
        """
        Input:
            gmagent: Game Agent
            gameState: Current Game State
            returngoalPosition: not return position if it's False

        Output:
            Simulate the opponents' escape plan path to the boundries

        IDEA:
            It can be implemented into Offensive Agent and used when the opponents' ghosts are
            within 2 steps away.
        """
        ##get ghost position
        ###get own position
        ####call A-star..
        gmagent = self
        myPos = gmagent.getCurrentObservation().getAgentPosition(gmagent.index)
        enemies = [gameState.getAgentState(i) for i in gmagent.getOpponents(gameState)]
        enemiesPacman = [a.getPosition() for a in enemies if a.isPacman and a.getPosition() != None]
        if len(enemiesPacman) == 0:
            return []
        walls = gameState.getWalls()
        height = walls.height
        width = walls.width
        walls = walls.asList()

        half_position = (
            int(gameState.data.layout.width / 2 - gmagent.red * (-1)), int(gameState.data.layout.height / 2))

        X_half = gameState.data.layout.width / 2 - gmagent.red * (-1)
        X_opp = enemiesPacman[0][0]
        X_mypos = myPos[0]

        X_diff = (X_opp - X_half) * (X_opp - X_mypos)

        half_position = (min(max(X_mypos - (1 if X_opp >= X_mypos else -1), 3), width - 3), 1)

        if X_diff < 0 or X_opp == X_mypos:
            half_position = (
                int(gameState.data.layout.width / 2 - gmagent.red * (-1)), int(gameState.data.layout.height / 2))

        while (gameState.hasWall(half_position[0], half_position[1])):
            half_position = (half_position[0], half_position[1] - 1)
        startPos = enemiesPacman[0]
        avoidPos = [myPos]
        goalPositions = [(half_position[0], height_position) for height_position in range(1, height - 1) if
                         not gameState.hasWall(half_position[0], height_position)]

        X = min(width / 4, 3)
        Y = min(height / 4, 3)

        for posX in range(int(max(1, myPos[0] - X)), int(min(width, myPos[0] + X))):
            for posY in range(int(max(0, myPos[1] - Y)), int(min(height, myPos[1] + Y))):
                if not gameState.hasWall(posX, posY):
                    if (abs(posX - myPos[0]) + abs(posY - myPos[1])) <= 2:
                        avoidPos.append((posX, posY))
                    if (posX, posY) in goalPositions:
                        goalPositions.remove((posX, posY))

        if len(goalPositions) == 0:
            return None
        return self.aStarSearch(gameState, goalPositions=goalPositions, startPosition=startPos,
                                avoidPositions=avoidPos, returngoalPosition=False)

    def InterceptOpponents(self, gameState):
        gmagent = self
        if len(gmagent.observationHistory) < 10:
            return None
        walls = gameState.getWalls().asList()
        actions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
        actionVectors = [(int(Actions.directionToVector(action)[0]), int(Actions.directionToVector(action)[1])) for
                         action in actions]

        myPos = gmagent.getCurrentObservation().getAgentPosition(gmagent.index)
        index_use = [i for i in gmagent.getOpponents(gameState)]
        enemies = [gameState.getAgentState(i) for i in gmagent.getOpponents(gameState)]
        enemiesPacman = [a.getPosition() for a in enemies if a.isPacman and a.getPosition() != None]
        if len(enemiesPacman) == 0:
            return None  ##JUST IN CASE
        currentPos = enemiesPacman[0]

        index_use = index_use[0]

        possiblePositions = [(currentPos[0] + vector[0], currentPos[1] + vector[1]) for vector, action in
                             zip(actionVectors, actions)]
        legalPositions = [position for position in possiblePositions if position not in walls]
        if len(gmagent.observationHistory) > 2:
            myPreState = gmagent.observationHistory[-2]
            # myPrePos = myPreState.getAgentPosition(gmagent.index)
            Preenemies = [myPreState.getAgentState(i) for i in gmagent.getOpponents(myPreState) if i == index_use]
            Prechaser = [a.getPosition() for a in Preenemies if a.isPacman and a.getPosition() != None]
            if len(Prechaser) == 0:
                return None
            if (gmagent.getMazeDistance(myPos, enemiesPacman[0]) < gmagent.getMazeDistance(myPos,
                                                                                           Prechaser[0])): return None
        ##========##
        # if len(legalPositions)>2:
        #    return None
        ##========##
        OpponentPath = self.EscapePathForOpponent(gameState)
        if len(OpponentPath) == 0:
            return None
        lenParameter = min(6, len(OpponentPath))
        OpponentPath = OpponentPath[:lenParameter]

        avoidPos = [enemiesPacman[0]]
        goalPos = []
        startPos = myPos

        for act in OpponentPath:
            actVec = actionVectors[actions.index(act)]
            currentPos = (currentPos[0] + actVec[0], currentPos[1] + actVec[1])
            goalPos.append(currentPos)

        AlternativePath, GoalPosition = self.aStarIntercept(gameState, goalPos, startPos, avoidPos,
                                                            enemiesPacman[0], returngoalPosition=True)
        if len(AlternativePath) <= lenParameter:
            # print("MINDFUL", GoalPosition)
            return GoalPosition
        return None

    def aStarIntercept(self, gameState, goalPositions, startPosition, avoidPositions, OriginalPosition,
                       returngoalPosition=False, returnCost=False):
        """
        Similar with aStar but adding heuristic function with penalties

        """
        gmagent = self
        walls = gameState.getWalls()
        width = walls.width
        height = walls.height
        walls = walls.asList()
        if startPosition == None:
            startPosition = gmagent.getCurrentObservation().getAgentPosition(gmagent.index)

        actions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
        actionVectors = [(int(Actions.directionToVector(action)[0]), int(Actions.directionToVector(action)[1])) for
                         action in actions]
        currentPosition, currentPath, currentCost = startPosition, [], 0

        queue = util.PriorityQueueWithFunction(lambda entry: entry[2] +  # Total cost so far
                                                             # width * height if entry[0] in avoidPositions else 0 +  # Avoid enemy locations like the plague
                                                             min(gmagent.getMazeDistance(entry[0],
                                                                                         endPosition) - 2 * gmagent.getMazeDistance(
                                                                 endPosition, OriginalPosition) for endPosition in
                                                                 goalPositions)
                                               )
        # No Revisits
        visited = set([currentPosition])

        while currentPosition not in goalPositions:
            possiblePositions = [((currentPosition[0] + vector[0], currentPosition[1] + vector[1]), action) for
                                 vector, action in zip(actionVectors, actions)]
            legalPositions = [(position, action) for position, action in possiblePositions if position not in walls]
            for position, action in legalPositions:
                if position not in visited:
                    visited.add(position)
                    AvoidValue = 0
                    if position in avoidPositions:
                        AvoidValue = width * height
                    queue.push((position, currentPath + [action], currentCost + 1 + AvoidValue))
            if queue.isEmpty():  ##Just in case
                return None
            else:
                currentPosition, currentPath, currentCost = queue.pop()
        if returnCost:
            return currentPath, currentPosition, currentCost

        if returngoalPosition:
            return currentPath, currentPosition

        else:
            return currentPath
        return currentPath