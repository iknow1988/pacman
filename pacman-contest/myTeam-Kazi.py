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
import random, time, util
from game import Directions
import game
import distanceCalculator
import random, time, util, sys
random.seed(42)
from util import nearestPoint
from pacman import GameState
import numpy as np
import operator

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
							 first = 'OffensiveMAB', second = 'TestDefensiveReflexAgent'):
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
				dist = self.getMazeDistance(self.start,pos2)
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

	def getRemainingScareTime(gameState, agentIndex):
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
		if myState.isPacman: features['onDefense'] = 0

		# Computes distance to invaders we can see
		enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
		invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
		features['numInvaders'] = len(invaders)
		if len(invaders) > 0:
			dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
			# print dists
			features['invaderDistance'] = min(dists)

		if action == Directions.STOP: features['stop'] = 1
		rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
		if action == rev: features['reverse'] = 1

		return features

	def getWeights(self, gameState, action):
		return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}

class MAB(CaptureAgent):

	def registerInitialState(self, gameState):
		CaptureAgent.registerInitialState(self, gameState)
		self.distancer.getMazeDistances()
		self.QValues = util.Counter()
		self.newQValues = util.Counter()
		self.lastState = None
		self.lastAction = None
		self.epsilon = 0.05
		self.discount = 0.8
		self.alpha = 0.2
		self.start = gameState.getAgentPosition(self.index)
		self.lastGameState = None
		self.MAX_FOOD_CARRYING = 3
		self.count = 0
		self.food_carrying = 0

	def chooseAction(self, gameState):
		actions = gameState.getLegalActions(self.index)
		return random.choice(actions)

	def getSuccessor(self, gameState, action):
		successor = gameState.generateSuccessor(self.index, action)
		pos = successor.getAgentState(self.index).getPosition()
		if pos != nearestPoint(pos):
			return successor.generateSuccessor(self.index, action)
		else:
			return successor

	def evaluate(self, gameState, action):
		features = self.getFeatures(gameState, action)
		weights = self.getWeights(gameState, action)
		return features * weights

	def getFeatures(self, gameState, action):
		features = util.Counter()
		successor = self.getSuccessor(gameState, action)
		features['successorScore'] = self.getScore(successor)
		return features

	def getWeights(self, gameState, action):
		return {'successorScore': 1.0}

class OffensiveMAB(MAB):

	def getFeatures(self, gameState, action):
		features = util.Counter()
		successor = self.getSuccessor(gameState, action)
		foodList = self.getFood(successor).asList()
		features['successorScore'] = -len(foodList)
		if len(foodList) > 0:  # This should always be True,  but better safe than sorry
			myPos = successor.getAgentState(self.index).getPosition()
			minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
			if self.food_carrying < self.MAX_FOOD_CARRYING:
				features['distanceToFood'] = minDistance
			else:
				features['distanceToFood'] = self.getMazeDistance(self.start,myPos)

		myPos = successor.getAgentState(self.index).getPosition()
		enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
		inRange = filter(lambda x: not x.isPacman and x.getPosition() != None, enemies)
		if len(inRange) > 0:
			positions = [agent.getPosition() for agent in inRange]
			closest = min(positions, key=lambda x: self.getMazeDistance(myPos, x))
			closestDist = self.getMazeDistance(myPos, closest)
			if closestDist <= 5:
				features['distanceToGhost'] = closestDist
			else:
				features['distanceToGhost'] = 0

		return features

	def getWeights(self, gameState, action):
		return {'successorScore': 100, 'distanceToFood': -1, 'distanceToGhost': 2}

	def chooseAction(self, gameState):
		self.food_carrying = gameState.getAgentState(self.index).numCarrying
		# print(self.food_carrying)
		action = self.getBestAction(gameState)
		nextGameState, reward = self.takeBestAction(gameState, action)
		self.updateQValue(gameState.getAgentState(self.index).getPosition(), action, nextGameState, reward)
		self.count = self.count + 1
		return action

	def getQValue(self, state, action):
		return self.QValues[(state, action)]

	def computeValueFromQValues(self, gameState):
		state = gameState.getAgentState(self.index).getPosition()
		qvalues = [self.getQValue(state, action) for action in gameState.getLegalActions(self.index)]
		actions = gameState.getLegalActions(self.index)
		for a in actions:
			self.QValues[(state,a)] = self.evaluate(gameState,a)
			qvalues.append(self.QValues[(state,a)])
		return max(qvalues)

	def getValue(self, gameState):
		return self.computeValueFromQValues(gameState)

	def computeActionFromQValues(self, gameState):
		legalActions = gameState.getLegalActions(self.index)
		legalActions.remove('Stop')
		state = gameState.getAgentState(self.index).getPosition()
		if not len(legalActions):
			return None
		QValue = -1e10
		temp = {}
		for legalAction in legalActions:
			QValueTemp = self.getQValue(state, legalAction)
			temp[legalAction] = QValueTemp
			if QValueTemp > QValue:
				action = legalAction
				QValue = QValueTemp
		print action, QValue,sorted(temp.items(), key=operator.itemgetter(1), reverse= True)
		return action

	def getPolicy(self, gameState):
		if self.count:
			return self.computeActionFromQValues(gameState)
		else:
			action = None
			actions = gameState.getLegalActions(self.index)
			for a in actions:
				self.QValues[(gameState.getAgentState(self.index).getPosition(), a)] = self.evaluate(gameState, a)
			pos = gameState.getAgentState(self.index).getPosition()
			QValue = -1e10
			for a in actions:
				if (pos, a) in self.QValues:
					QValueTemp = self.QValues[(pos, a)]
				else:
					QValueTemp = -1e10
				if QValueTemp > QValue:
					action = a
					QValue = QValueTemp
			return action

	def getBestAction(self, gameState):
		legalActions = gameState.getLegalActions(self.index)
		action = None
		if not len(legalActions):
			return action
		randomAction = util.flipCoin(self.epsilon)
		if randomAction:
			action = random.choice(legalActions)
		else:
			action = self.getPolicy(gameState)
		return action

	def takeBestAction(self, gameState, action):
		self.lastState = self
		self.lastAction = action
		self.lastGameState = gameState
		self.count = self.count + 1
		successor = self.getSuccessor(gameState, action)
		reward = successor.getScore() - gameState.getScore()
		state = successor.getAgentPosition(self.index)
		return successor, reward

	def update(self):
		print("here")
	def updateQValue(self, state, action, nextGameState, reward):
		curQValue = self.getQValue(state, action)
		self.QValues[(state, action)] = (1 - self.alpha) * curQValue + self.alpha * (reward+ self.discount * self.getValue(nextGameState))
		# print(self.QValues)
