# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 02:58:28 2018

@author: Administrator
"""

# baselineTeam.py
# ---------------
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


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html
from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
import distanceCalculator
import random, time, util, sys
from util import nearestPoint

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveReflexAgent', second = 'TestDefensiveReflexAgent'):
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

class DummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)

    '''
    Your initialization code goes here, if you need any.
    '''
    # we need to define initial state

  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)

    '''
    You should change this in your own agent.
    '''

    return random.choice(actions)


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









##================BASELINE OFFENSIVE AGENT===================##
class OffensiveReflexAgent(TestReflexCaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()    
    features['successorScore'] = -len(foodList)#self.getScore(successor)

    # Compute distance to the nearest food

    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'distanceToFood': -1}







##================DEFENSIVE AGENT===================##
class TestDefensiveReflexAgent(TestReflexCaptureAgent):
   
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
        #print(self.scare)
    elif self.scare==1:
        self.scare=CapsuleMonitor(self, gameState,self.scare)
        #print(self.scare)
    #print(myState)
    #print(type(myState))
    """Sactions=successor.getLegalActions(self.index)
    #print(Sactions)
    for act in Sactions:
        suc=successor.generateSuccessor(self.index, act)
        sucPos = suc.getAgentState(self.index)
        #print(self.getMazeDistance(sucPos.getPosition(),(1,1)))
    print("==================")"""
    
    half_position=(gameState.data.layout.width/2,gameState.data.layout.height/2)
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

    if self.target_position==None and len(alerting)==0:
        kmeans_positions=kmeans(self.getFoodYouAreDefending(successor))
        key_pos=[half_position]
        for pos,k in kmeans_positions:
            if k > 3:
                key_pos.append(pos)
        print(key_pos)    
    
        self.target_position=key_pos[int(random.uniform(0,len(key_pos)))]
    
    if self.target_position!=None:
        
        print("inside loop", self.target_position)
        features['targetPosition']=self.getMazeDistance(self.target_position,myPos)

    return features

  def getWeights(self, gameState, action):
    ##Priority Different
    return {'invaderDistance': -1000, 'numInvaders': -1000, 'MissingFood':-100,'targetPosition':-100,
            'onDefense': 10, 'stop': -10, 'reverse': -2}







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
     
        while(1):
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
                    if dis2<dis_close:
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
        food_pos=0
        for food in prefoodlist:
          if food !=missingfoods:
            cur_dist=gmagent.getMazeDistance(missingfoods,food)
            if cur_dist<dist:
              dist=cur_dist
              food_pos=food
        ret_list.append((food_pos,x))
    return ret_list 
  

def CapsuleMonitor(gmagent, gameState, scare, last=40):
    #print("In CapsuleMonitor")
    if scare==0:
        index=-1
        preind=index-1
        if len(gmagent.observationHistory)>2:
            curCaplist=gmagent.getCapsulesYouAreDefending(gmagent.observationHistory[index])
            preCaplist=gmagent.getCapsulesYouAreDefending(gmagent.observationHistory[preind])
            #print(curCaplist,preCaplist)
            if(len(preCaplist)-len(curCaplist)==1):
                return 1
    if scare==1 and len(gmagent.observationHistory)>2:
        if gameState.getAgentPosition(gmagent.index)==gmagent.observationHistory[0].getAgentPosition(gmagent.index):
            return 0
    return scare













