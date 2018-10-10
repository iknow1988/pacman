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
from game import Directions
import game
import distanceCalculator
import random, time, util, sys
import pickle
random.seed(42)
from util import nearestPoint
from pacman import GameState
import numpy as np
import operator
import os
from game import Actions
import sys
import pickle
#################
# Team creation #
#################


def createTeam(firstIndex, secondIndex, isRed, first = 'OffensiveQAgent', second = 'DefensiveQAgent'):
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
            offset = 1
        else:
            offset = -1
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


class OffensiveQAgent(ApproximateQAgent):

    def __init__(self, index, **args):
        ApproximateQAgent.__init__(self, index, **args)
        self.filename = "kazi_offensive.agent.weights"
        self.weights = util.Counter()
        if os.path.exists(self.filename):
            with open(self.filename, "rb") as f:
                self.weights = pickle.load(f)
        else:
            # self.weights = util.Counter({
            # 	'bias': -4.921244227779046,
            # 	'distanceToGhost': -0.023246811873140483,
            # 	'targetPosition': -1.9415049256734194,
            # 	'successorScore': -0.05505235521062358
            # })
            # self.weights = util.Counter({
            # 	'bias': -4.733133320570293,
            # 	'distanceToGhost': -1.1450613632412818,
            # 	'targetPosition': 0.15716149651465722,
            # 	'successorScore': -0.05505235521062358
            # })
            # self.weights = util.Counter({
            #     'distanceToGhost': -0.40802311580190537,
            #     'bias': -3.3717795430968844,
            #     'numOfGhosts': -1.1099380879088678,
            #     'successorScore': 0.9169816334644033,
            #     'targetPosition': -3.618340692571109,
            #     'distanceToInvader': 0.015331274121942063
            # })
            self.weights = util.Counter({
                'distanceToGhost': -0.09789200941236537,
                'bias': -4.218878193773446,
                'numOfGhosts': 0.3426987251631774,
                'successorScore': 0.9169816334644033,
                'targetPosition': -8.6353361911183,
                'distanceToInvader': 0.4645185484656794
            })
        self.freeTimerToEatFood = 3
        self.target_position = None
        self.carryLimit = 10
        self.alternativePath = []

    def final(self, state):
        with open(self.filename, 'wb') as f:
            pickle.dump(self.weights, f)
        CaptureAgent.final(self, state)

    def registerInitialState(self, gameState):
        ApproximateQAgent.registerInitialState(self, gameState)
        self.target_position = min(self.entrances, key=lambda x: self.getMazeDistance(gameState.getAgentState(self.index).getPosition(), x))
        if(len(self.getFood(gameState).asList()))>0:
            self.target_position = max(self.getFood(gameState).asList(),
                                       key=lambda x: self.getMazeDistance(gameState.getAgentState(self.index).getPosition(),
                                                                          x))

    def getFeatures(self, state, action):
        myCurrentState = state.getAgentState(self.index)
        myCurrentPosition = myCurrentState.getPosition()
        successor = self.getSuccessor(state, action)
        foodList = self.getFood(successor).asList()
        myNextState = successor.getAgentState(self.index)
        myNextPosition = myNextState.getPosition()
        ghosts = self.getGhosts(successor)
        invaders = self.getInvaders(successor)
        features = util.Counter()
        minDistanceToGhost = 0.0
        minDistanceToInvader = 0.0
        minDistanceToFood = 0.0
        distancesToGhosts = [0.0]
        distancesToInvaders = [0.0]

        if len(ghosts) > 0:
            self.carryLimit = 10
            distancesToGhosts = [self.getMazeDistance(myNextPosition, a.getPosition()) for a in ghosts]
            if not self.isOpponentScared(successor) and min(distancesToGhosts) <= 1 and myNextState.isPacman:
                minDistanceToGhost = min(distancesToGhosts) * 1.0

        if len(ghosts) > 0:
            distancesToGhosts = [self.getMazeDistance(myNextPosition, a.getPosition()) for a in ghosts]
            minDist = min(distancesToGhosts)
            if minDist > 5:
                self.carryLimit = self.max_score
            else:
                self.carryLimit = 10

        if len(invaders) > 0:
            distancesToInvaders = [self.getMazeDistance(myNextPosition, a.getPosition()) for a in invaders]
            if min(distancesToInvaders) <= 1 and not myNextState.isPacman:
                minDistanceToInvader = -min(distancesToInvaders) * 1.0

        # eaten a food, giving another food to eat
        if myCurrentPosition == self.target_position:
            self.target_position = min(foodList, key=lambda x: self.getMazeDistance(myNextPosition, x))

        # eaten everything go back home
        if len(foodList) == 1:
            self.target_position = min(self.entrances, key=lambda x: self.getMazeDistance(myNextPosition, x))

        # reached carry limit go back to home
        if myNextState.numCarrying >= self.carryLimit:
            self.target_position = min(self.entrances, key=lambda x: self.getMazeDistance(myNextPosition, x))

        # End of time and go back to home
        timeLeft = state.data.timeleft * 1.0 / state.getNumAgents()
        target = min(self.entrances, key=lambda x: self.getMazeDistance(myNextPosition, x))
        dist = self.getMazeDistance(myNextPosition, target)

        if timeLeft - dist < 1.0:
            self.target_position = target

        features["bias"] = 1.0
        features['numOfGhosts'] = len(ghosts)
        features['distanceToGhost'] = minDistanceToGhost
        features['distanceToInvader'] = minDistanceToInvader
        features['targetPosition'] = self.getMazeDistance(myNextPosition, self.target_position) * 1.0 / self.gridSize

        # self.debugDraw(self.target_position, (1, 0, 0), clear=True)

        return features

    def chooseAction(self, gameState):
        action = None
        if len(self.alternativePath) > 0:
            action = self.alternativePath.pop()
            if len(self.alternativePath) == 0:
                self.target_position = self.entrances[int(random.uniform(0, len(self.entrances)))]
        elif not self.CheckIfAgentStucking(gameState):
            actions = gameState.getLegalActions(self.index)
            action = self.computeActionFromQValues(gameState)
        else:
            actions, goalDestination = self.FindAlternativeFood(gameState)
            if actions:
                self.alternativePath = actions
                action = self.alternativePath.pop()
            else:
                actions = gameState.getLegalActions(self.index)
                action = random.choice(actions)

        self.doAction(gameState, action)  # from Q learning agent

        return action

    def anyFoodLeft(self, state):
        if len(self.getFood(state).asList()) > 0:
            return True
        else:
            return False

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

    def isOpponentScared(self, state):
        scared = False
        enemies = [state.getAgentState(i) for i in self.getOpponents(state)]
        for a in enemies:
            if not a.isPacman:
                if a.scaredTimer > self.freeTimerToEatFood:
                    scared = True
                    break
        return scared

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

    def getGhosts(self, state):
        enemies = [state.getAgentState(i) for i in self.getOpponents(state)]
        ghosts = [a for a in enemies if not a.isPacman and a.getPosition()]

        return ghosts

    def getInvaders(self, state):
        enemies = [state.getAgentState(i) for i in self.getOpponents(state)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition()]

        return invaders

    def computeActionFromQValues(self, gameState):
        actions = gameState.getLegalActions(self.index)
        actions.remove('Stop')
        values = [self.getQValue(gameState, a) for a in actions]
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        best = random.choice(bestActions)

        return best


class DefensiveQAgent(ApproximateQAgent):

    def __init__(self, index, **args):
        ApproximateQAgent.__init__(self, index, **args)

    def registerInitialState(self, gameState):
        ApproximateQAgent.registerInitialState(self, gameState)
        self.filename = "kazi_defensive.agent.weights"
        self.weights = util.Counter()
        self.carryLimit = 10
        self.target_position = None
        self.alternativePath = []
        if os.path.exists(self.filename):
            with open(self.filename, "r") as f:
                self.weights = pickle.load(f)
        else:
            # self.weights = util.Counter({
            #     'bias': -4.952049116175205,
            #     'missingFoodDistance': -8.12789923148146,
            #     'distanceToEntrance': -6.622842281335308,
            #     'scaredState': 1.1355706099852891,
            #     'isPacman': -0.22433502168640782,
            #     'numOfInvaders': 1.0752513842357354,
            #     'invaderDistance': -18.058777676262988
            # })
            # self.weights = util.Counter({
            # 	'bias': -5.844442694708891,
            # 	'missingFoodDistance': -0.9547637059066729,
            # 	'distanceToEntrance': -3.2695878086524433,
            # 	'scaredState': 1.7559998314945628,
            # 	'isPacman': 0.023955081944477805,
            # 	'numOfInvaders': 1.5056732292001267,
            # 	'invaderDistance': -28.036091500393667
            # })
            # self.weights = util.Counter({
            #     'bias': -4.8865635555343685,
            #     'missingFoodDistance': -12.525337056313884,
            #     'distanceToEntrance': -8.909959384844386,
            #     'scaredState': 1.0681305090282578,
            #     'isPacman': 0.22489953810723354,
            #     'numOfInvaders': 1.3442286774963974,
            #     'invaderDistance': -27.499290588725366
            # })
            # self.weights = util.Counter({
            #     'bias': -6.1932559736488155,
            #     'missingFoodDistance': -12.651776463751023,
            #     'distanceToEntrance': -8.935734302258002,
            #     'scaredState': 1.0681305090282578,
            #     'isPacman': 0.22784574059864363,
            #     'numOfInvaders': 0.8385944376041887,
            #     'invaderDistance': -27.497792993317265
            # })
            self.weights = util.Counter({
                'bias': -4.586074248564161,
                'missingFoodDistance': -15.81901072573371,
                'distanceToEntrance': -10.971157338028735,
                'scaredState': 1.0681305090282578,
                'isPacman': 4.874879766979384,
                'numOfInvaders': 1.2587071647319488,
                'invaderDistance': -27.295889489781025
            })

    def final(self, state):
        with open(self.filename, 'wb') as f:
            pickle.dump(self.weights, f)
        CaptureAgent.final(self, state)

    def getFeatures(self, state, action):
        features = util.Counter()
        myPosition = state.getAgentState(self.index).getPosition()
        successor = self.getSuccessor(state, action)
        newState = successor.getAgentState(self.index)
        newPos = newState.getPosition()
        invaders = self.getInvaders(state)
        ghosts = self.getGhosts(successor)
        missingFoods = self.getMissingFoods(state)

        if not self.target_position or self.target_position == myPosition:
            if not self.target_position or len(self.getFoodYouAreDefending(state).asList()) == 0:
                if len(self.getFoodYouAreDefending(state).asList()) > 0 :
                    closest = min(self.getFoodYouAreDefending(state).asList(),
                                  key=lambda x: self.getMazeDistance(myPosition, x))
                    self.target_position = closest
                else:
                    self.target_position = self.minDistantEntrance

            elif self.getScore(state) >= 10 or \
                    state.getAgentState(self.index).numCarrying >= self.carryLimit:
                entrances = self.entrances
                distances = util.Counter()
                for entrance in entrances:
                    dist = 0
                    for food in self.getFoodYouAreDefending(state).asList():
                        dist = dist + self.getMazeDistance(food, entrance)
                    distances[entrance] = dist
                keyPos = min(distances, key=distances.get)
                self.target_position = keyPos
            elif len(self.getFoodYouAreDefending(state).asList()) < 5 and len(self.getFoodYouAreDefending(state).asList()) > 0:
                entrances = self.entrances
                distances = util.Counter()
                for entrance in entrances:
                    closest = min(self.getFoodYouAreDefending(state).asList(), key=lambda x: self.getMazeDistance(entrance, x))
                    distances[entrance] = self.getMazeDistance(entrance, closest)
                self.target_position = min(distances, key=distances.get)

            elif len(missingFoods) < 1:
                self.carryLimit = 3
                if len(self.getFood(successor).asList()) > 0:
                    foods = self.getFood(successor).asList()
                    closest = min(foods, key=lambda x: self.getMazeDistance(newPos, x))
                    self.target_position = closest
            elif len(self.getFoodYouAreDefending(state).asList()) > 5:
                if len(self.getFood(successor).asList()) > 0:
                    foods = self.getFood(successor).asList()
                    closest = min(foods, key=lambda x: self.getMazeDistance(newPos, x))
                    self.target_position = closest
            else:
                entrances = self.entrances
                distances = util.Counter()
                for entrance in entrances:
                    dist = 0
                    distances[entrance] = min(self.getFoodYouAreDefending(state).asList(), key=lambda x: self.getMazeDistance(newPos, x))
                keyPos = min(distances, key=distances.get)
                self.target_position = keyPos

        if self.lastState:
            lastStatefoodList = self.getFood(self.lastState).asList()
            if self.target_position in lastStatefoodList and self.target_position not in self.getFood(state).asList():

                self.target_position = min(self.entrances, key=lambda x: self.getMazeDistance(newPos, x))

        features['invaderDistance'] = 0.0
        distanceToInvaders = [0]
        distanceToGhosts = [0]
        if len(invaders) > 0:
            distanceToInvaders = [self.getMazeDistance(newPos, a.getPosition()) for a in invaders]
            if not newState.isPacman:
                features['invaderDistance'] = min(distanceToInvaders) * 1.0 / self.gridSize

        features["isPacman"] = 0.0
        if not newState.isPacman:
            self.alternativePath = []

        if len(ghosts) > 0:
            distanceToGhosts = [self.getMazeDistance(newPos, a.getPosition()) for a in ghosts]
        if newState.isPacman and min(distanceToGhosts)<=1:
            features["isPacman"] = -1.0 * min(distanceToGhosts)

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
        features["bias"] = 1.0
        features['numOfInvaders'] = len(invaders)

        # self.debugDraw(self.target_position, (0, 1, 0), clear=True)

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
        reward = self.getRecoveredFoodCount(state, lastState)
        reward -= len(self.getInvaders(state)) - len(self.getInvaders(lastState))
        if self.target_position != state.getAgentState(self.index).getPosition():
            reward -= 1
        distancePosition = self.getMazeDistance(state.getAgentState(self.index).getPosition(), lastState.getAgentState(self.index).getPosition())
        if distancePosition > 1:
            reward -= distancePosition * 1.0/self.gridSize
        return reward

    def getInvaders(self, state):
        enemies = [state.getAgentState(i) for i in self.getOpponents(state)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition()]

        return invaders

    def getGhosts(self, state):
        enemies = [state.getAgentState(i) for i in self.getOpponents(state)]
        invaders = [a for a in enemies if not a.isPacman and a.getPosition()]

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
        # return 'Stop'
        actions = gameState.getLegalActions(self.index)
        actions.remove('Stop')
        values = [self.getQValue(gameState, a) for a in actions]

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        best = random.choice(bestActions)

        return best

    def chooseAction(self, gameState):
        actions = gameState.getLegalActions(self.index)
        if len(self.alternativePath) > 0:
            action = self.alternativePath.pop()
            if len(self.alternativePath) == 0:
                self.target_position = self.entrances[int(random.uniform(0, len(self.entrances)))]
        elif not gameState.getAgentState(self.index).isPacman and self.CloggingOpponent(gameState):
            action = 'Stop'

        elif gameState.getAgentState(self.index).isPacman and self.CheckIfAgentStucking(gameState):
            actions, goalDestination = self.FindAlternativeFood(gameState)
            if actions:
                self.alternativePath = actions
                action = self.alternativePath.pop()
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
                                           avoidPositions=avoidPositions, returngoalPosition=False, returnCost=True)
        if Cost > width * height:
            return True

        return False