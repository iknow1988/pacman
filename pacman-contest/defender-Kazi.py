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
import pickle
random.seed(42)
from util import nearestPoint
from pacman import GameState
import numpy as np
import operator
import os
from game import Actions


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

	def registerInitialState(self, gameState):
		self.start = gameState.getAgentPosition(self.index)
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
		self.entrances = entrances
		self.gridSize = self.walls .width * self.walls .height
		self.initialDefendingFoodCount = len(self.getFoodYouAreDefending(gameState).asList())

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
		if util.flipCoin(self.epsilon):
			action = random.choice(actions)
		else:
			action = self.computeActionFromQValues(gameState)

		self.doAction(gameState, action)  # from Q learning agent
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


class DefensiveApproximateQAgent(ApproximateQAgent):

	def __init__(self, index, **args):
		ApproximateQAgent.__init__(self, index, **args)
		self.filename = "defensive.agent.weights"
		self.weights = util.Counter()
		if os.path.exists(self.filename):
			with open(self.filename, "rb") as f:
				self.weights = pickle.load(f)
				print self.weights

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
		missingFoods = self.getMissingFoods(state)

		features["bias"] = 1.0
		features['numOfInvaders'] = len(invaders)
		if len(invaders) > 0:
			distanceToInvaders = [self.getMazeDistance(newPos, a.getPosition()) for a in invaders]
			features['invaderDistance'] = min(distanceToInvaders) * 1.0 / self.gridSize
			if newState.scaredTimer > 0:
				features['scaredState'] = (min(distanceToInvaders) - newState.scaredTimer) * 1.0 / self.gridSize
			else:
				features['scaredState'] = 0

		elif len(missingFoods) > 0:
			dist_miss = [self.getMazeDistance(newPos, a) for a,i in missingFoods]
			features['missingFoodDistance'] = min(dist_miss) * 1.0 / self.gridSize
		elif len(missingFoods) == 0:
			maxDistance = max([self.getMazeDistance(newPos, food) for food in self.getFoodYouAreDefending(state).asList()])
			features['maximumFoodDistance'] = maxDistance * 1.0 / self.gridSize
		else:
			entrances = self.entrances
			pos = entrances[int(random.uniform(0, len(entrances)))]
			features['distanceToEntrance'] = self.getMazeDistance(pos, newPos) * 1.0 / self.gridSize

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
		distancePosition = self.getMazeDistance(state.getAgentState(self.index).getPosition(),
												lastState.getAgentState(self.index).getPosition())
		if distancePosition > 1:
			reward -= distancePosition * -1.0/self.gridSize

		return reward

	def getInvaders(self, state):
		enemies = [state.getAgentState(i) for i in self.getOpponents(state)]
		invaders = [a for a in enemies if a.isPacman and a.getPosition()]

		return invaders

	def getRecoveredFoodCount(self, state, lastState):
		return len(self.getFoodYouAreDefending(state).asList()) - len(self.getFoodYouAreDefending(lastState).asList())

	def getMissingFoods(self, gameState, steps=6):
		counter = min((len(self.observationHistory) - 1), steps)
		missingFoods = []
		for x in range(1, counter + 1):
			index = -x
			preind = index - 1
			currentFoodList = self.getFoodYouAreDefending(self.observationHistory[index]).asList()
			previousFoodList = self.getFoodYouAreDefending(self.observationHistory[preind]).asList()
			missing = [i for i in previousFoodList if i not in currentFoodList]
			if len(missing) != 0:
				missing = missing[0]
				dist = 9999999
				food_pos = previousFoodList[0]
				for food in previousFoodList:
					if food != missing:
						cur_dist = self.getMazeDistance(missing, food)
						if cur_dist < dist:
							dist = cur_dist
							food_pos = food
				missingFoods.append((food_pos, x))

		return missingFoods

	def computeActionFromQValues(self, gameState):
		actions = gameState.getLegalActions(self.index)
		actions.remove('Stop')
		values = [self.getQValue(gameState, a) for a in actions]

		maxValue = max(values)
		bestActions = [a for a, v in zip(actions, values) if v == maxValue]
		best = random.choice(bestActions)
		return best