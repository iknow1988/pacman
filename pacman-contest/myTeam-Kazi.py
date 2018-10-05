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


class OffensiveQAgent(ApproximateQAgent):

	def __init__(self, index, **args):
		ApproximateQAgent.__init__(self, index, **args)
		self.carryLimit = 10
		self.PowerTimer = 3
		self.filename = "offensive.agent.weights"
		self.weights = util.Counter()
		if os.path.exists(self.filename):
			with open(self.filename, "rb") as f:
				self.weights = pickle.load(f)
		print "Initial", self.weights

	def final(self, state):
		with open(self.filename, 'wb') as f:
			pickle.dump(self.weights, f)
		print self.weights
		CaptureAgent.final(self, state)

	# def getFeatures(self, state, action):
	# 	currentGameState = state.getAgentState(self.index)
	# 	myPosition = currentGameState.getPosition()
	# 	teammatePositions = [state.getAgentPosition(teammate) for teammate in self.getTeam(state)]
	# 	capsulePos = state.getRedCapsules() if state.isOnRedTeam(self.index) else state.getBlueCapsules()
	# 	otherTeam = state.getBlueTeamIndices() if state.isOnRedTeam(self.index) else state.getRedTeamIndices()
	# 	successor = self.getSuccessor(state, action)
	# 	foodList = self.getFood(successor).asList()
	# 	nextState = successor.getAgentState(self.index)
	# 	nextPosition = nextState.getPosition()
	# 	ghosts = self.getGhosts(state)
	#
	# 	features = util.Counter()
	# 	features["bias"] = 1.0
	# 	features['numOfGhosts'] = len(ghosts)
	# 	features['successorScore'] = -1.0 * (len(foodList)-2)/self.max_score
	#
	# 	minDistanceToFood = 0.0
	# 	goToHome = 0.0
	# 	if len(foodList) > 2:
	# 		minDistanceToFood = min([self.getMazeDistance(nextPosition, food) for food in foodList])
	# 	else:
	# 		goToHome = min([self.getMazeDistance(nextPosition, entrance) for entrance in self.entrances])
	# 	features['distanceToFood'] = minDistanceToFood * 1.0 / self.gridSize
	# 	features['distanceToHome'] = goToHome * 1.0 / self.gridSize
	#
	# 	# if len(ghosts) > 0:
	# 	# 	distanceToInvaders = [self.getMazeDistance(nextPosition, a.getPosition()) for a in ghosts]
	# 	# 	features['distanceToGhost'] = min(distanceToInvaders) * 1.0 / self.gridSize
	# 	# 	features['freeToEat'] = 0.0
	# 	# 	for a in ghosts:
	# 	# 		if a.scaredTimer > 0:
	# 	# 			features['freeToEat'] = (a.scaredTimer - min(distanceToInvaders)) * 1.0 / self.gridSize
	#
	# 	# features['goToHome'] = -self.getMazeDistance(self.minDistantEntrance, nextPosition)* 1.0 / self.gridSize
	#
	# 	return features

	def getFeatures(self, state, action):
		myPrevState = state.getAgentState(self.index)
		myPrePos = myPrevState.getPosition()
		otherTeam = state.getBlueTeamIndices() if state.isOnRedTeam(self.index) else state.getRedTeamIndices()
		successor = self.getSuccessor(state, action)
		foodList = self.getFood(successor).asList()
		myState = successor.getAgentState(self.index)
		myPos = myState.getPosition()
		ghosts = self.getGhosts(state)
		features = util.Counter()
		features["bias"] = 1.0
		features['successorScore'] = -(len(foodList)-2)*1.0 / self.max_score
		features['numOfGhosts'] = len(ghosts)
		distanceToInvaders = [0.0]
		if len(ghosts) > 0:
			distanceToInvaders = [self.getMazeDistance(myPos, a.getPosition()) for a in ghosts]
		features['distanceToGhost'] = min(distanceToInvaders) * 1.0 / self.gridSize
		checkall = 0
		for index in otherTeam:
			otherAgentState = state.data.agentStates[index]
			if otherAgentState.scaredTimer > self.PowerTimer:  # power capsule
				checkall += 1
			if otherAgentState.isPacman:
				continue
			ghostPosition = otherAgentState.getPosition()
			if not ghostPosition:
				continue
			if otherAgentState.scaredTimer <= self.PowerTimer:
				features["#-of-ghosts-1-step-away"] = int(myPos in Actions.getLegalNeighbors(ghostPosition, self.walls))
		if myPrevState.numCarrying == 0:
			self.carryLimit = len(foodList)
		if (features["#-of-ghosts-1-step-away"] and myPrevState.numCarrying != 0) or (
				state.data.timeleft / 1. / state.getNumAgents() / 2. < self.walls.width):
			self.carryLimit = myPrevState.numCarrying if myPrevState.numCarrying != 0 else 2
		if len(foodList) > 0:  # This should always be True,  but better safe than sorry
			dis = sorted([self.getMazeDistance(myPos, food) for food in foodList])
			minDistance = dis[0]
		else:
			minDistance = 0
		features['distanceToFood'] = float(minDistance)
		back_home = False
		if checkall != len(otherTeam):
			if myPrevState.numCarrying >= self.carryLimit:
				back_home = True
		if len(foodList) == 0:
			back_home = True
		if back_home:
			features['back-home'] = - 1. * self.getMazeDistance(self.start, myPos) / (self.walls.width * 5.)
			features['distanceToFood'] = 0.
		features['distanceToFood'] /= 1. * self.gridSize
		# features['numOfCarrying'] = 1.0 * successor.getAgentState(self.index).numCarrying
		return features

	def observationFunction(self, state):
		if self.lastState:
			reward = self.getRewards(state, self.lastState)
			self.observeTransition(self.lastState, self.lastAction, state, reward)

		return CaptureAgent.observationFunction(self, state)

	def getRewards(self, state, lastState):
		# if self.getFoodCount(state, lastState)>0:
		# 	reward = 1.0 * self.getFoodCount(state, lastState)
		# else:
		# 	reward = 1.0 * self.getFoodCount(state, lastState)/self.max_score
		# reward += 1.0 * state.getAgentState(self.index).numCarrying / self.max_score
		reward = (state.getScore() - lastState.getScore())*1.0 / self.max_score
		# print (state.getScore() - lastState.getScore())* 1.0 / self.max_score
		reward -= 1.0
		# print "reward shaping", (self.getScore(state) - self.max_score)/ self.max_score
		# distancePosition = self.getMazeDistance(state.getAgentState(self.index).getPosition(),
		# 										lastState.getAgentState(self.index).getPosition())
		# if distancePosition > 1:
		# 	# print reward
		# 	reward -= ((distancePosition * 1.0) / self.gridSize)
		# 	# print ((distancePosition * 1.0) / self.gridSize)
		# # print reward
		return reward

	def getFoodCount(self, state, lastState):
		return len(self.getFood(lastState).asList()) - len(self.getFood(state).asList())

	def getGhosts(self, state):
		enemies = [state.getAgentState(i) for i in self.getOpponents(state)]
		invaders = [a for a in enemies if not a.isPacman and a.getPosition()]

		return invaders

	def computeActionFromQValues(self, gameState):
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
		print action, features, weights * features

		return weights * features

	def computeActionFromQValues(self, gameState):
		actions = gameState.getLegalActions(self.index)
		actions.remove('Stop')
		values = [self.getQValue(gameState, a) for a in actions]

		maxValue = max(values)
		bestActions = [a for a, v in zip(actions, values) if v == maxValue]
		best = random.choice(bestActions)
		print best, maxValue, zip(actions, values)
		return best


class DefensiveQAgent(ApproximateQAgent):

	def __init__(self, index, **args):
		ApproximateQAgent.__init__(self, index, **args)
		self.filename = "defensive.agent.weights"
		self.weights = util.Counter()
		with open(self.filename, "rb") as f:
			self.weights = pickle.load(f)
		print "Initial", self.weights

	def final(self, state):
		# with open(self.filename, 'wb') as f:
		# 	pickle.dump(self.weights, f)
		# print "Updated", self.weights
		ApproximateQAgent.final(self, state)

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
		# print "FOOD LEFT: ", len(self.getFoodYouAreDefending(state).asList())
		if self.target_position == newPos:
			entrances = self.entrances
			distances = util.Counter()
			for entrance in entrances:
				dist = 0
				for food in self.getFoodYouAreDefending(state).asList():
					dist = dist + self.getMazeDistance(food, entrance)
				distances[entrance] = dist
			keyPos = min(distances, key=distances.get)
			self.target_position = keyPos
		# self.target_position = entrances[int(random.uniform(0, len(entrances)))]


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