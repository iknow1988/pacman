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
import sys
sys.path.append('teams/poison/')

def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveApproximateQAgent', second='DefensiveQAgent'):
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
        self.actions_counter=0
        self.atacking=True
       

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

    def isInvaderDead(self, state):
        enemies = [state.getAgentState(i) for i in self.getOpponents(state)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition()]
        positions = [agent.getPosition() for agent in invaders]
        myPos = state.getAgentState(self.index).getPosition()
        if len(positions)>0:
            closest = min(positions, key=lambda x: self.getMazeDistance(myPos, x))
            closestDist = self.getMazeDistance(myPos, closest)
            if closestDist==1:
                return True
            else:
                return False
        else:
            return False

class ApproximateQAgentDG(CaptureAgent):

    def __init__(self, index, epsilon=0.05, alpha=0.8, gamma=0.8, **args):
        CaptureAgent.__init__(self, index)
        self.epsilon = float(epsilon)
        self.alpha = float(alpha)
        self.discount = float(gamma)
        self.qValues = util.Counter()
        self.lastState = None
        self.lastAction = None
        self.start = None
        self.actions_counter=0
        self.atacking=False

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)

        # get middle
        self.walls = gameState.getWalls()
        """
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
        """

        self.gridSize = self.walls.width * self.walls.height
        self.initialDefendingFoodCount = len(self.getFoodYouAreDefending(gameState).asList())
        self.key_positions=self.keyPositions(self,gameState)

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
        if self.atacking:
            self.actions_counter+=1
            
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

    def keyPositions(self,gmagent, gameState):
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
            
        validPositions=[pos for pos in [half_position, FirstquaterPosition, ThirdquaterPosition] if pos[0]>=0 and pos[1]>=0]
      #start_position=gmagent.observationHistory[0].getAgentPosition(gmagent.index)
        ##we dont want our key positions to be too close to the starting position
        validPositions=[pos for pos in validPositions if gmagent.getMazeDistance(pos,self.start)>=5]
        
        return validPositions


class DefensiveApproximateQAgent(ApproximateQAgentDG):

    def __init__(self, index, **args):
        ApproximateQAgent.__init__(self, index, **args)
        self.filename = "./teams/Poison/defensive.agent.weights"
        self.weights = util.Counter()
        self.MAX_FOOD_CARRYING=2
        self.foodToSecure=[]
        self.atacking=False

        if os.path.exists(self.filename):
            with open(self.filename, "rb") as f:
                self.weights = pickle.load(f)
        
        #if len(self.weights)==0:
        #    self.weights['distanceToFoodToSecure']=1
        #    self.weights['distanceToFoodToAttack']=0.5

    def final(self, state):
        with open(self.filename, 'wb') as f:
            pickle.dump(self.weights, f)
        CaptureAgent.final(self, state)

    def getFeatures(self, state, action):
        features = util.Counter()

        
        #get state attributes from the environment
        currentFoodListDefending=self.getFoodYouAreDefending(state).asList()
        currentPosition=state.getAgentState(self.index).getPosition()
        
        successor = self.getSuccessor(state, action)
        newState = successor.getAgentState(self.index)
        newPos = newState.getPosition()
        
        
        
        lastFoodEaten=[]
        
        prevState=None
        if len(self.observationHistory)>0:
            prevState=self.getPreviousObservation()
           
        prevFoodListDefending=[]
        

        
        

        ########### DEFAULT FEATURES ###################
        #### bias feature
        features["bias"] = 1.0
        #### 

        ##### Defense=1  / offensive = 0
        #if not state.getAgentState(self.index).isPacman:
            #features['onDefense'] = 1
        #else:
            #features['onDefense'] = 0
        #### 

        ##### Number of invaders 
        invaders = self.getInvaders(state)
        features['numOfInvaders'] = len(invaders)
        
        ##### invader distance
        if len(invaders) > 0:
            dists = [self.getMazeDistance(
                newPos, a.getPosition()) for a in invaders]
            closestDist = min(dists)
            if closestDist <= 5:
                features['invaderDistance'] = -float(closestDist)/self.gridSize
        #### 
       
        ##### Food to secure or key points
        # Compute distance to the nearest food. 
        if len(self.foodToSecure) == 0:
            self.foodToSecure=self.key_positions[0]

        if self.lastState!=None:
            # calculate the last food eaten in previous state
            prevFoodListDefending=self.getFoodYouAreDefending(self.lastState).asList()
            lastFoodEaten=list(set(prevFoodListDefending) - set(currentFoodListDefending))
            
            # calculate the next probable move
            if len(lastFoodEaten)!=0:
                self.foodToSecure=lastFoodEaten[-1]
                # if last food eaten has not change then keep the last value
            else:
                lastFoodEaten=self.foodToSecure
        
        foodListToAttack = self.getFood(successor).asList()
        if not self.isInvaderDead(state) and not self.atacking:
            minDistanceToSecure = self.getMazeDistance(newPos, self.foodToSecure)
            features['distanceToFoodToSecure'] = -float(minDistanceToSecure) / self.gridSize
            
        else:
            # Attack and back home+
            
            if self.actions_counter<=15:
                minDistanceToAttack = min([self.getMazeDistance(newPos, food) for food in foodListToAttack])
                features['distanceToFoodToAttack'] = -float(minDistanceToAttack)/self.gridSize
                self.atacking=True
            else:
                self.atacking=False
                self.actions_counter=0 # reset the action counter

        if self.atacking:
            print "atacking"    
        else:
            print self.foodToSecure

        print self.weights
        print action,features
        #### 

        #### scared feature
        #if newState.scaredTimer > 0:
        #    features['scaredState'] = (min(closestDist) - newState.scaredTimer) * 1.0 / self.gridSize
        #else:
        #    features['scaredState'] = 0
        ####
        


            
        
        """
        # condition of return to home
        if state.getAgentState(self.index).isPacman:
            if state.getAgentState(self.index).numCarrying >= self.MAX_FOOD_CARRYING:
                minDistanceToSecure = min([self.getMazeDistance(newPos, food) for food in self.foodToSecure])
                features['returnHome'] =  -float(minDistanceToSecure)/ (self.walls.width * 5.)
        
            enemies = [successor.getAgentState(i)
                    for i in self.getOpponents(successor)]
            inRange = filter(
                lambda x: not x.isPacman and x.getPosition() != None, enemies)
            if len(inRange) > 0:
                positions = [agent.getPosition() for agent in inRange]
                closest = min(
                    positions, key=lambda x: self.getMazeDistance(newPos, x))
                closestDist = self.getMazeDistance(newPos, closest)
                if closestDist <= 5:
                    features['distanceToGhostToAvoid'] = float(closestDist) /self.gridSize
        #######
        """
        return features

    def observationFunction(self, state):
        if self.lastState:
            reward = (state.getScore() - self.lastState.getScore()) # dominant reward
            
            reward+=self.getRewards(state,self.lastState) # reward shape
            
            self.observeTransition(self.lastState, self.lastAction, state, reward)

        return CaptureAgent.observationFunction(self, state)

    
    def isInvaderDead(self, state):
        enemies = [state.getAgentState(i) for i in self.getOpponents(state)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition()]
        positions = [agent.getPosition() for agent in invaders]
        myPos = state.getAgentState(self.index).getPosition()
        if len(positions)>0:
            closest = min(positions, key=lambda x: self.getMazeDistance(myPos, x))
            closestDist = self.getMazeDistance(myPos, closest)
            if closestDist==1:
                return True
            else:
                return False
        else:
            return False

    def getInvaders(self, state):
        enemies = [state.getAgentState(i) for i in self.getOpponents(state)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition()]

        return invaders

    def getRecoveredFoodCount(self, state, lastState):
        return len(self.getFoodYouAreDefending(state).asList()) - len(self.getFoodYouAreDefending(lastState).asList())

    def getMissingFood(self,gmagent, gameState, steps=6):
        """
        This function gives the information of missing food within previous n(default=3) steps
        
        This function takes gameState as input and return a list [((1,3),1), ((1,4),2), (1,5),3)] 
        this means the closest food to the food was eaten in the recent one step is at position (1,3), 
        and the closest food to the food that is eaten in the previous 2 step is at (1,4),
        and the closest food to the food that is eaten in the previous 3 step is at (1,5)
        that is to say the opponents pacman may be move from (1,2)->(1,3)->(1,4) accordingly. 
        
        """
        
        itera=min((len(gmagent.observationHistory)-1), steps)
        ret_list=[]
        for x in range(1,itera+1):
            index=-x
            preind=index-1
            curfoodlist=gmagent.getFoodYouAreDefending(gmagent.observationHistory[index]).asList()
            prefoodlist=gmagent.getFoodYouAreDefending(gmagent.observationHistory[preind]).asList()
            missingfoods=[i for i in prefoodlist if i not in curfoodlist]
            if len(missingfoods)!=0:
                missingfoods=missingfoods[0]
                dist=9999999
                food_pos=prefoodlist[0]
                for food in prefoodlist:
                    if food !=missingfoods:
                        cur_dist=gmagent.getMazeDistance(missingfoods,food)
                        if cur_dist<dist:
                            dist=cur_dist
                            food_pos=food
                ret_list.append((food_pos,x))
        return ret_list 


    def computeActionFromQValues(self, gameState):
        actions = gameState.getLegalActions(self.index)
        actions.remove('Stop')
        values = [self.getQValue(gameState, a) for a in actions]

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        best = random.choice(bestActions)
        return best

    def getRewards(self, state, lastState):
        reward=0
        reward -= 1 #penalty for every second
        
        reward = self.getRecoveredFoodCount(state, lastState)
        reward -= len(self.getInvaders(state)) - len(self.getInvaders(lastState))
        """
        lastPosition=lastState.getAgentState(self.index).getPosition()
        currentPosition=state.getAgentState(self.index).getPosition()
        lastMovDistance = self.getMazeDistance(lastPosition,currentPosition)
        if lastMovDistance > 1:
            reward -= float(lastMovDistance)/self.gridSize

        # penalizes for being pacman and loose food
        if (state.getAgentState(self.index).isPacman) and self.getRecoveredFoodCount(state, lastState) <0:
            reward-=1
        """
        return reward

    

    
class DefensiveQAgent(ApproximateQAgent):

    def __init__(self, index, **args):
        ApproximateQAgent.__init__(self, index, **args)
        self.filename = "teams/Poison/defensive.agent.weights"
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
        
        invaders = list(self.getInvaders(state))
        missingFoods = self.getMissingFoods(state)
        features["bias"] = 1.0
        features['numOfInvaders'] = len(invaders)

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
        features['invaderDistance'] = 0.0
        distanceToInvaders = [0]
        if len(invaders) > 0:
            """
            distanceToInvaders = [self.getMazeDistance(newPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(distanceToInvaders) * 1.0 / self.gridSize
            features["isPacman"] = 0.0
            if newState.isPacman:
                distanceToThreats=self.getDistanceOpponentsGhosts(successor)
                if  distanceToThreats<= 3:
                    features["isPacman"] = -1.0 * distanceToThreats/ self.gridSize
                    print "Danger!"
                else:
                    print "Not in Danger!"
            """
            distanceToInvaders = [self.getMazeDistance(newPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(distanceToInvaders) * 1.0 / self.gridSize
            features["isPacman"] = 0.0
            if newState.isPacman:
                features["isPacman"] = -1.0 * min(distanceToInvaders) * 1.0 / self.gridSize

        features['scaredState'] = 0.0
        if newState.scaredTimer > 0:
            features['scaredState'] = (min(distanceToInvaders) - newState.scaredTimer) * 1.0 / self.gridSize

        
        foodListToAttack = self.getFood(successor).asList()
        goHome=False
        if successor.getAgentState(self.index).numCarrying > len(foodListToAttack)*0.2 or len(foodListToAttack)==0:goHome=True 
        

        if len(self.getInvadersNoPosition(state))>0  or goHome:
            dist_miss = 0.0
            if len(missingFoods) > 0:
                self.target_position = missingFoods[0][0]
                for pos, i in missingFoods:
                    dist_miss += self.getMazeDistance(pos, newPos)
            features['missingFoodDistance'] = dist_miss * 1.0 / self.gridSize

            minDistEntrance = self.getMazeDistance(newPos, self.target_position)
            features['distanceToEntrance'] = minDistEntrance * 1.0 / self.gridSize
        else:
            ### 

            closest = min(foodListToAttack, key=lambda x: self.getMazeDistance(newPos, x))
            self.target_position=closest
            minDistEntrance = self.getMazeDistance(newPos, self.target_position)
            features['distanceToEntrance'] = minDistEntrance * 1.0 / self.gridSize
            self.atacking=True
            
          
        self.debugDraw(self.target_position, (0,1,0), clear=True)
            
        return features

    def observationFunction(self, state):
        if self.lastState:
            reward = self.getRewards(state, self.lastState)
            self.observeTransition(self.lastState, self.lastAction, state, reward)

        return CaptureAgent.observationFunction(self, state)

    def getRewards(self, state, lastState):
        reward=0
        reward -= 1 # default penalty
        reward += self.getRecoveredFoodCount(state, lastState) # doing good job defending
        reward += (state.getScore() - self.lastState.getScore()) # rewarded if winning, penalizes if getting our food go away
        reward -= len(self.getInvaders(state)) - len(self.getInvaders(lastState)) # penalizes if letting invaders in our field
        reward+=state.getAgentState(self.index).numCarrying # reward for eating food of the oponent
        # penalty if got killed
        distancePosition = self.getMazeDistance(state.getAgentState(self.index).getPosition(), lastState.getAgentState(self.index).getPosition())
        if distancePosition > 1:
            reward -= distancePosition * 1.0/self.gridSize 
            #reward-=1
            

        return reward

    def getDistanceOpponentsGhosts(self, state):
        enemies = [state.getAgentState(i) for i in self.getOpponents(state)]
        ghosts = [a for a in enemies if not a.isPacman and a.getPosition()!=None]
        positions = [agent.getPosition() for agent in ghosts]
        myPos = state.getAgentState(self.index).getPosition()
        if len(positions)>0:
            closest = min(positions, key=lambda x: self.getMazeDistance(myPos, x))
            closestDist = self.getMazeDistance(myPos, closest)
            if closestDist <= 5:
                return closestDist
            else:
                return 10
        else:
            return 10

    def getInvaders(self, state):
        enemies = [state.getAgentState(i) for i in self.getOpponents(state)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition()]

        return invaders

    def getInvadersNoPosition(self, state):
        enemies = [state.getAgentState(i) for i in self.getOpponents(state)]
        invaders = [a for a in enemies if a.isPacman]

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


class OffensiveApproximateQAgent(ApproximateQAgent):

    def __init__(self, index, **args):
        ApproximateQAgent.__init__(self, index, **args)

        self.MAX_FOOD_CARRYING =3

        self.filename = "./teams/Poison/offensive.agent.weights"
        self.weights = util.Counter()
       
        if os.path.exists(self.filename):
            with open(self.filename, "rb") as f:
                self.weights = pickle.load(f)
        
            

    def final(self, state):
        with open(self.filename, 'wb') as f:
            pickle.dump(self.weights, f)
        CaptureAgent.final(self, state)

    def getFeatures(self, state, action):
        """
        Get features used for state evaluation.
        """
        # initialization
        features = util.Counter()
        successor = self.getSuccessor(state, action)
        features["bias"] = 1.0

        #previous states and positions
        prevFood = self.getFood(state)
        myPrevState = state.getAgentState(self.index)
        myPrePos = myPrevState.getPosition()
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        
        
        foodList = self.getFood(successor).asList()
        self.food_carrying=state.getAgentState(self.index).numCarrying

        if self.food_carrying >= self.MAX_FOOD_CARRYING:
            features['successorScore'] = state.getScore()
        else:
            features['successorScore'] = -len(foodList)/self.gridSize

        # Compute distance to the nearest food. If it's in mode safe it shouldnt look for food.
        if len(foodList) > 0 and self.food_carrying < self.MAX_FOOD_CARRYING:
            minDistance = min([self.getMazeDistance(myPos, food)
                               for food in foodList])
            features['distanceToFood'] = -float(minDistance)/self.gridSize
        
        # Compute distance to closest ghost
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
                features['distanceToGhost'] = float(closestDist) /self.gridSize
                #if self.debug: self.debugDraw(myPos, self.goal_color, clear=True)
           
        

        features.divideAll(10.0)

        
        return features


    def observationFunction(self, state):
        if self.lastState:
            reward = (state.getScore() - self.lastState.getScore()) # dominant reward
            reward+=self.getRewards(state,self.lastState) # reward shape # reward shape
            self.observeTransition(self.lastState, self.lastAction, state, reward)

        return CaptureAgent.observationFunction(self, state)


    def computeActionFromQValues(self, gameState):
        actions = gameState.getLegalActions(self.index)
        actions.remove('Stop')
        values = [self.getQValue(gameState, a) for a in actions]

        #print("Values:",values)
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        
        best = random.choice(bestActions)
        return best

    def getRewards(self, state, lastState):
        reward=0
        reward-=1 
        
        
        #penalization if got killed without food
        myPos = state.getAgentState(self.index).getPosition()
        myPrevPos= lastState.getAgentState(self.index).getPosition()
        myCurFoodCarrying = state.getAgentState(self.index).numCarrying
        myPrevFoodCarrying= lastState.getAgentState(self.index).numCarrying
        if myPrevPos!=self.start and myPos==self.start:
            #reward-=1
            if (myCurFoodCarrying<myPrevFoodCarrying):
                reward-=myCurFoodCarrying-myPrevFoodCarrying
        
        reward+=myCurFoodCarrying
        

        
        return reward

 
