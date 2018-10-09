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

def createTeam(firstIndex, secondIndex, isRed,
			   first = 'OffensiveQAgent', second = 'DefensiveQAgent'):
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
		# if util.flipCoin(self.epsilon):
		# 	action = random.choice(actions)
		# else:
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


class OffensiveQAgent(ApproximateQAgent):

	def __init__(self, index, **args):
		ApproximateQAgent.__init__(self, index, **args)
		self.filename = "test.offensive.agent.weights"
		self.weights = util.Counter()
		if os.path.exists(self.filename):
			with open(self.filename, "rb") as f:
				self.weights = pickle.load(f)
		# print "initial", self.weights
		self.carryLimit = 10
		self.freeTimerToEatFood = 3
		self.target_position_offensive = None
		self.foodTryCount = 0

	def final(self, state):
		with open(self.filename, 'wb') as f:
			pickle.dump(self.weights, f)
		# print "Updated", self.weights
		CaptureAgent.final(self, state)

	def getFeatures(self, state, action):
		myPrevState = state.getAgentState(self.index)
		myPrePos = myPrevState.getPosition()
		successor = self.getSuccessor(state, action)
		foodList = self.getFood(successor).asList()
		myState = successor.getAgentState(self.index)
		myPos = myState.getPosition()
		ghosts = self.getGhosts(successor)
		features = util.Counter()

		minDistanceToInvaders = 0.0
		minDistanceToFood = 0.0
		distToGhostsList = [0.0]
		if myPrevState.numCarrying == 0:
			self.carryLimit = 10
		if not self.target_position_offensive or myPrePos == self.target_position_offensive:
			closest = None
			if not self.target_position_offensive:
				if(len(foodList)>0):
					closest = max(foodList, key=lambda x: self.getMazeDistance(myPrePos, x))
			elif len(ghosts) > 0:
				distToGhostsList = [self.getMazeDistance(myPrePos, a.getPosition()) for a in ghosts]
				minDistance = min(distToGhostsList) * 1.0
				if minDistance > 6 :
					if (len(foodList) > 0):
						closest = min(foodList, key=lambda x: self.getMazeDistance(myPrePos, x))
				else:
					closest = sorted(self.entrances, key=lambda x: self.getMazeDistance(myPrePos, x))[int(len(self.entrances)/2)]
			else:
				if (len(foodList) > 0):
					closest = min(foodList, key=lambda x: self.getMazeDistance(myPrePos, x))
			if closest:
				self.target_position_offensive = closest
			else:
				self.target_position_offensive = self.minDistantEntrance

		if len(ghosts) > 0:
			distToGhostsList = [self.getMazeDistance(myPos, a.getPosition()) for a in ghosts]
			if not self.isOpponentScared(successor) and min(distToGhostsList) <= 2:
				minDistanceToInvaders = min(distToGhostsList) * 1.0
				self.carryLimit = myPrevState.numCarrying if myPrevState.numCarrying != 0 else 2

		if not self.isOpponentScared(successor):
			if myPrevState.numCarrying >= self.carryLimit:
				self.target_position_offensive = min(self.entrances, key=lambda x: self.getMazeDistance(myPos, x))
			elif minDistanceToInvaders and minDistanceToInvaders <=2 and myPrevState.isPacman:
				self.target_position_offensive = min(self.entrances, key=lambda x: self.getMazeDistance(myPos, x))
			elif len(foodList) == 1:
				self.target_position_offensive = min(self.entrances, key=lambda x: self.getMazeDistance(myPos, x))

		features["bias"] = 1.0
		features['successorScore'] = -len(foodList)*1.0 / self.max_score
		features['distanceToGhost'] = minDistanceToInvaders
		features['targetPosition'] = self.getMazeDistance(myPos, self.target_position_offensive) * 1.0 / self.gridSize
		self.debugDraw(self.target_position_offensive, (1, 0, 0), clear=True)
		return features

	def isOpponentScared(self,state):
		scared = False
		enemies = [state.getAgentState(i) for i in self.getOpponents(state)]
		for a in enemies:
			if not a.isPacman:
				if a.scaredTimer > self.freeTimerToEatFood:
					scared = True
					break
		return scared

	def observationFunction(self, state):
		if self.lastState:
			reward = self.getRewards(state, self.lastState)
			self.observeTransition(self.lastState, self.lastAction, state, reward)
		return CaptureAgent.observationFunction(self, state)

	def getRewards(self, state, lastState):
		foodEaten = self.getFoodCount(state, lastState)
		reward = 0
		# if foodEaten > 0:
		# 	reward = self.getFoodCount(state, lastState)
		reward += state.getScore() - lastState.getScore()
		reward -= 1


		return reward

	def getFoodCount(self, state, lastState):
		return len(self.getFood(state).asList()) - len(self.getFood(lastState).asList())

	def getGhosts(self, state):
		enemies = [state.getAgentState(i) for i in self.getOpponents(state)]
		invaders = [a for a in enemies if not a.isPacman and a.getPosition()]

		return invaders

	def computeActionFromQValues(self, gameState):
		# return 'Stop'
		actions = gameState.getLegalActions(self.index)
		actions.remove('Stop')
		values = [self.getQValue(gameState, a) for a in actions]
		maxValue = max(values)
		bestActions = [a for a, v in zip(actions, values) if v == maxValue]
		best = random.choice(bestActions)
		# print(best,maxValue, zip(actions, values))
		return best

	def getQValue(self, state, action):
		weights = self.getWeights()
		features = self.getFeatures(state, action)
		# print "------------->",action, features, weights * features

		return weights * features

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
															 min(util.manhattanDistance(entry[0], endPosition) for
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

	def FindAlternativeFood(self, gameState, returngoalPosition=True):
		gmagent = self
		myPos = gmagent.getCurrentObservation().getAgentPosition(gmagent.index)
		enemies = [gameState.getAgentState(i) for i in gmagent.getOpponents(gameState)]
		chasers = [a.getPosition() for a in enemies if not a.isPacman and a.getPosition() != None]
		walls = gameState.getWalls()

		foodList = gmagent.getFood(gameState).asList()

		height = walls.height
		width = walls.width
		walls = walls.asList()
		half_position = (int(gameState.data.layout.width / 2 - gmagent.red), int(gameState.data.layout.height / 2))
		while (gameState.hasWall(half_position[0], half_position[1])):
			half_position = (half_position[0], half_position[1] - 1)

		possibleGoalPositions = foodList

		avoidPos = []
		X = min(width / 4, 3)
		Y = min(height / 4, 3)

		for chaser in chasers:
			for posX in range(int(max(1, chaser[0] - X)), int(min(width, chaser[0] + X))):
				for posY in range(int(max(0, chaser[0] - Y)), int(min(height, chaser[0] + Y))):
					if not (gameState.hasWall(posX, posY) or ((abs(posX - chaser[0]) + abs(posY - chaser[1]))) <= 4):
						avoidPos.append((posX, posY))
		##Here return a list and the position
		return self.aStarSearch(gameState, goalPositions=possibleGoalPositions, startPosition=myPos, avoidPositions=avoidPos,
						   returngoalPosition=returngoalPosition)


class DefensiveQAgent(ApproximateQAgent):

	def __init__(self, index, **args):
		ApproximateQAgent.__init__(self, index, **args)
		self.filename = "test.defensive.agent.weights"
		self.weights = util.Counter()
		self.carryLimit = 3
		if os.path.exists(self.filename):
			with open(self.filename, "r") as f:
				self.weights = pickle.load(f)
		print "initial", self.weights

	def final(self, state):
		with open(self.filename, 'w') as f:
			pickle.dump(self.weights, f)
		print "Updated", self.weights
		ApproximateQAgent.final(self, state)

	def getFeatures(self, state, action):
		features = util.Counter()
		myPosition = state.getAgentState(self.index).getPosition()
		successor = self.getSuccessor(state, action)
		newState = successor.getAgentState(self.index)
		newPos = newState.getPosition()
		self.inLoopCount[newPos] = self.inLoopCount[newPos] + 1
		invaders = self.getInvaders(state)
		ghosts = self.getInvaders(state)
		missingFoods = self.getMissingFoods(state)
		features["bias"] = 1.0
		features['numOfInvaders'] = len(invaders)
		# print "FOOD LEFT: ", len(self.getFoodYouAreDefending(state).asList())
		if self.target_position == myPosition:
			if self.getScore(state) >= self.carryLimit or state.getAgentState(self.index).numCarrying >= self.carryLimit:
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
				if(len(self.getFood(successor).asList())>0):
					foods = self.getFood(successor).asList()
					closest = min(foods, key=lambda x: self.getMazeDistance(newPos, x))
					self.target_position = closest

		features['invaderDistance'] = 0.0
		distanceToInvaders = [0]
		distanceToGhosts = [0]
		if len(invaders) > 0:
			distanceToInvaders = [self.getMazeDistance(newPos, a.getPosition()) for a in invaders]
			if not newState.isPacman:
				features['invaderDistance'] = min(distanceToInvaders) * 1.0 / self.gridSize

		features["isPacman"] = 0.0
		if len(ghosts) > 0:
			distanceToGhosts = [self.getMazeDistance(newPos, a.getPosition()) for a in ghosts]
		if newState.isPacman and min(distanceToGhosts)<3:
			features["isPacman"] = -2.0 * min(distanceToInvaders) / self.gridSize

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
		self.debugDraw(self.target_position, (0, 1, 0), clear=True)
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
		# print reward
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
		actions = gameState.getLegalActions(self.index)
		actions.remove('Stop')
		values = [self.getQValue(gameState, a) for a in actions]

		maxValue = max(values)
		bestActions = [a for a, v in zip(actions, values) if v == maxValue]
		best = random.choice(bestActions)
		return best