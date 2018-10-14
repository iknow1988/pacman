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
        for action in actions:
            if not self.RunForestCheckDeadAlley(gameState, action):
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
                if a.scaredTimer > 5:
                    scared = True
                    break

        return scared


class OffensiveQAgent(ApproximateQAgent):

    def __init__(self, index, **args):
        ApproximateQAgent.__init__(self, index, **args)
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
        # self.weights = util.Counter({
        #     'distanceToGhost': -0.7488480965816826,
        #     'bias': -2.5703680099396724,
        #     'numOfGhosts': 0.05322959640985453,
        #     'successorScore': 0.9169816334644033,
        #     'targetPosition': -8.768669849614495,
        #     'distanceToInvader': 0.4645185484656794
        # })
        self.weights = util.Counter({
            'distanceToGhost': -0.8276702904874571,
            'bias': -1.163443217229148,
            'numOfGhosts': -0.42320086673733487,
            'successorScore': 0.9169816334644033,
            'targetPosition': -8.855595889050168,
            'distanceToInvader': -1.59733990137347
        })
        self.freeTimerToEatFood = 5
        self.target_position = None
        self.carryLimit = self.max_score
        self.alternativePath = []

    def registerInitialState(self, gameState):
        ApproximateQAgent.registerInitialState(self, gameState)
        self.target_position = min(self.entrances, key=lambda x: self.getMazeDistance(gameState.getAgentState(self.index).getPosition(), x))
        if len(self.getFood(gameState).asList()) > 0:
            self.target_position = max(self.getFood(gameState).asList(),
                                       key=lambda x: self.getMazeDistance(gameState.getAgentState(self.index).getPosition(),
                                                                          x))
        self.carryLimit = self.max_score

    def getFeatures(self, state, action):
        myCurrentState = state.getAgentState(self.index)
        myCurrentPosition = myCurrentState.getPosition()
        successor = self.getSuccessor(state, action)
        foodList = self.getFood(state).asList()
        myNextState = successor.getAgentState(self.index)
        myNextPosition = myNextState.getPosition()
        ghosts = self.getGhosts(successor)
        invaders = self.getInvaders(successor)
        features = util.Counter()
        minDistanceToGhost = 0.0
        minDistanceToInvader = 0.0
        capsules = self.getCapsules(state)
        minDistantCapsule = 0
        if len(capsules) > 0:
            minDistantCapsule = max(capsules, key=lambda x: self.getMazeDistance(myNextPosition, x))

        if len(ghosts) > 0:
            distancesToGhosts = [self.getMazeDistance(myNextPosition, a.getPosition()) for a in ghosts]
            if not self.isOpponentScared(successor) and min(distancesToGhosts) <= 1 and myNextState.isPacman:
                minDistanceToGhost = min(distancesToGhosts) * 1.0
            minDist = min(distancesToGhosts)
            if minDist > 5 or self.isOpponentScared(successor):
                self.carryLimit = self.max_score
            else:
                self.carryLimit = 10

        # eaten a food, giving another food to eat
        if myCurrentPosition == self.target_position and len(foodList)>0:
            self.target_position = min(foodList, key=lambda x: self.getMazeDistance(myNextPosition, x))

        # eaten everything go back home
        if len(foodList) <= 2:
            self.target_position = min(self.entrances, key=lambda x: self.getMazeDistance(myNextPosition, x))

        # reached carry limit go back to home
        if not self.isOpponentScared(state) and len(ghosts) > 0 and myCurrentState.numCarrying >= self.carryLimit:
            self.target_position = min(self.entrances, key=lambda x: self.getMazeDistance(myNextPosition, x))

        # End of time and go back to home
        timeLeft = state.data.timeleft * 1.0 / state.getNumAgents()
        target = min(self.entrances, key=lambda x: self.getMazeDistance(myNextPosition, x))
        dist = self.getMazeDistance(myNextPosition, target)
        if timeLeft - dist < 2.0:
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
        if not self.isOpponentScared(state) and self.LoopBreakerMoniter(state):
            self.target_position = min(self.entrances, key=lambda x: self.getMazeDistance(myNextPosition, x))

        features["bias"] = 1.0
        features['numOfGhosts'] = len(ghosts)
        features['distanceToGhost'] = minDistanceToGhost
        features['distanceToInvader'] = minDistanceToInvader / 2
        features['targetPosition'] = self.getMazeDistance(myNextPosition, self.target_position) * 1.0 / self.gridSize

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
        b = actions[:]
        ghosts = self.getGhosts(gameState)
        myCurrentState = gameState.getAgentState(self.index)
        myCurrentPosition = myCurrentState.getPosition()
        if len(ghosts) > 0:
            distancesToGhosts = [self.getMazeDistance(myCurrentPosition, a.getPosition()) for a in ghosts]
            if not self.isOpponentScared(gameState) and min(distancesToGhosts) <= 7 and myCurrentState.isPacman:
                if len(actions) > 0:
                    actions = self.getSafeActions(gameState, actions)

        values = [self.getQValue(gameState, a) for a in actions]
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        best = random.choice(bestActions)

        return best

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


class DefensiveQAgent(ApproximateQAgent):

    def __init__(self, index, **args):
        ApproximateQAgent.__init__(self, index, **args)
        self.carryLimit = 10
        self.target_position = None
        self.alternativePath = []
        self.initialFoodListDefending = []
        self.filename = "test.defensive.agent.weights"
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
        # self.weights = util.Counter({
        #     'bias': -5.423708441572619,
        #     'missingFoodDistance': -16.364344164244894,
        #     'distanceToEntrance': -11.179064509728025,
        #     'scaredState': 3.4064141580923644,
        #     'isPacman': 4.874879766979384,
        #     'numOfInvaders': 0.8040635735423013,
        #     'invaderDistance': -27.198030068319607
        # })
        self.weights = util.Counter({
            'bias': -4.6186314590369335,
            'missingFoodDistance': -18.94233117234949,
            'distanceToEntrance': -12.05954391621563,
            'scaredState': 3.018512473332247,
            'isPacman': 3.1300453104184762,
            'numOfInvaders': 1.2477133541777359,
            'invaderDistance': -27.109942306890893
        })

    def registerInitialState(self, gameState):
        ApproximateQAgent.registerInitialState(self, gameState)
        self.target_position = self.minDistantEntrance
        self.alternativePath = []
        self.initialFoodListDefending = self.getFoodYouAreDefending(gameState).asList()

    def getFeatures(self, state, action):
        features = util.Counter()
        myCurrentState = state.getAgentState(self.index)
        myPosition = state.getAgentState(self.index).getPosition()
        successor = self.getSuccessor(state, action)
        newState = successor.getAgentState(self.index)
        newPos = newState.getPosition()
        invaders = self.getInvaders(successor)
        ghosts = self.getGhosts(successor)
        missingFoods = self.getMissingFoods(state)
        foodListDefending = self.getFoodYouAreDefending(state).asList()
        foodListToEat = self.getFood(state).asList()
        capsules = self.getCapsules(state)
        minDistantCapsule = 0
        if len(capsules) > 0:
            minDistantCapsule = min(capsules, key=lambda x: self.getMazeDistance(newPos, x))

        if not newState.isPacman:
            self.alternativePath = []

        # goal aware target for defenses
        if self.target_position == myPosition and not myCurrentState.isPacman:
            if len(foodListDefending) > 6:
                distances = []
                for entrance in self.entrances:
                    distances.append([self.getMazeDistance(food, entrance) for food in foodListDefending])
                bestLocations = [a for a, v in zip(self.entrances, distances) if v == min(distances)]
                self.target_position = random.choice(bestLocations)
            elif 0 < len(foodListDefending) <= 6:
                successor_food_clusters = self.kmeans(self.getFoodYouAreDefending(state), 2)
                best_food_cluster = max(successor_food_clusters,
                                        key=lambda item: item[1])[0]
                self.target_position = best_food_cluster
            else:
                self.target_position = min(self.entrances, key=lambda x: self.getMazeDistance(newPos, x))

        distanceToInvaders = [0]
        minDistanceToInvader = 0.0
        minDistanceToGhost = 0.0
        if len(invaders) > 0:
            distanceToInvaders = [self.getMazeDistance(newPos, a.getPosition()) for a in invaders]
            if not newState.isPacman:
                minDistanceToInvader = min(distanceToInvaders)

        if len(ghosts) > 0:
            distanceToGhosts = [self.getMazeDistance(newPos, a.getPosition()) for a in ghosts]
            if newState.isPacman and min(distanceToGhosts) <= 2 and not self.isOpponentScared(state):
                minDistanceToGhost = -1.0 * min(distanceToGhosts)

        # Opponent ate capsule so go to offensive
        if newState.scaredTimer > 0 and len(foodListToEat) > 2:
            self.target_position = min(foodListToEat, key=lambda x: self.getMazeDistance(newPos, x))

        # Team is losing and go offensive
        if self.getScore(state) != 0 and self.getScore(state) <= 5 and len(foodListDefending) > 6 and len(foodListToEat) > 2:
            self.target_position = min(foodListToEat, key=lambda x: self.getMazeDistance(newPos, x))

        # Eat capsule if it is near
        if myCurrentState.isPacman and len(capsules) > 0 and not self.isOpponentScared(state):
            if self.getMazeDistance(newPos, self.target_position) > self.getMazeDistance(newPos, minDistantCapsule):
                self.target_position = minDistantCapsule

        # teammate ate the food and update to new target (coordination)
        if myCurrentState.isPacman and self.lastState:
            foodInLastState = self.target_position in self.getFood(self.lastState).asList()
            foodNotInPresentState = self.target_position not in foodListToEat
            if foodInLastState and foodNotInPresentState and len(foodListToEat) > 2:
                self.target_position = min(foodListToEat, key=lambda x: self.getMazeDistance(newPos, x))

        # Notifying defenseive agent when opponent ate a food
        dist_miss = 0.0
        if len(missingFoods) > 0 and newState.scaredTimer == 0:
            self.target_position = missingFoods[0][0]
            for pos, i in missingFoods:
                dist_miss += self.getMazeDistance(pos, newPos)

        # reached carry limit go back to home
        if newState.isPacman and not self.isOpponentScared(state) and len(ghosts)>0 \
                and myCurrentState.numCarrying >= self.carryLimit:
            self.target_position = min(self.entrances, key=lambda x: self.getMazeDistance(newPos, x))
            
        # End of time and go back to home
        timeLeft = state.data.timeleft * 1.0 / state.getNumAgents()
        target = min(self.entrances, key=lambda x: self.getMazeDistance(newPos, x))
        dist = self.getMazeDistance(newPos, target)
        if newState.isPacman and (timeLeft - dist < 1.0):
            self.target_position = target

        # # eaten everything go back home
        if newState.isPacman and len(foodListToEat) <= 2:
            self.target_position = min(self.entrances, key=lambda x: self.getMazeDistance(newPos, x))

        features['scaredState'] = 0.0
        if newState.scaredTimer > 0:
            features['scaredState'] = (min(distanceToInvaders) - newState.scaredTimer) * 1.0 / self.gridSize
        features['invaderDistance'] = minDistanceToInvader * 1.0 / self.gridSize
        features["isPacman"] = minDistanceToGhost
        features['missingFoodDistance'] = dist_miss * 1.0 / self.gridSize
        minDistEntrance = self.getMazeDistance(newPos, self.target_position)
        features['distanceToEntrance'] = minDistEntrance * 1.0 / self.gridSize
        features["bias"] = 1.0
        features['numOfInvaders'] = len(invaders)

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
        ghosts = self.getGhosts(gameState)
        myCurrentState = gameState.getAgentState(self.index)
        myCurrentPosition = myCurrentState.getPosition()
        if myCurrentState.isPacman and len(ghosts) > 0:
            distancesToGhosts = [self.getMazeDistance(myCurrentPosition, a.getPosition()) for a in ghosts]
            if len(ghosts)>0 and not self.isOpponentScared(gameState) and min(distancesToGhosts) <= 5 and myCurrentState.isPacman:
                if len(actions) > 0:
                    actions = self.getSafeActions(gameState, actions)
        values = [self.getQValue(gameState, a) for a in actions]

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        best = random.choice(bestActions)

        return best

    def chooseAction(self, gameState):
        actions = gameState.getLegalActions(self.index)
        if not gameState.getAgentState(self.index).isPacman and self.CloggingOpponent(gameState):
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
        centers=[]
        if len(foodlist)==0:
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