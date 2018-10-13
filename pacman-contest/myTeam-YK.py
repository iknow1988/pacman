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
from game import Actions
import distanceCalculator
import random, time, util, sys
from util import nearestPoint
import capture

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveReflexAgent', second = 'GTDefensiveReflexAgent'):
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
  
    def getFeatures(self, gameState, action):
        
        CheckIfAgentStucking(self, gameState)
        print(CheckIfAgentStucking(self, gameState))
        
        half_position=(int(gameState.data.layout.width/2-self.red),int(gameState.data.layout.height/2))
        while(gameState.hasWall(half_position[0],half_position[1])):
            half_position=(half_position[0],half_position[1]-1)    
        
        
        
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()
        if self.target_position==myPos:
            self.target_position=None
        ##Update self.scare
        missingfoodinf=getMissingFood(self, gameState)
        
        ##========TESTING CLOGGING=========##
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        chaser = [a.getPosition() for a in enemies if a.isPacman and a.getPosition() != None]
        
        features['Clog']=0
        if len(chaser) > 0:
            if CloggingOpponent(self, gameState, returngoalPosition = False) == True and action == Directions.STOP :
                features['Clog']=1000000000
        
    
        ##====A-star Example====:
        key_pos=keyPositions(self, gameState)#key_pos is a list
        
        Path, goalPosition=aStarSearch(gmagent=self, gameState=gameState, goalPositions=key_pos, returngoalPosition= True)
        
        
        Path2=EscapePath(gmagent=self, gameState=gameState, returngoalPosition=False)
        
        goal_path, goal_pos=FindAlternativeFood(self, gameState, returngoalPosition=True)
        if action == goal_path[0]:
            features['goalpath']=100000

        
        ##====END OF A-star Example====
        
        
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
      return {'invaderDistance': -100000, 'numInvaders': -100000, 'MissingFood':-1000,'targetPosition':-1000000,
              'onDefense': 10000, 'stop': -10, 'reverse': -2, 'Clog': 100000000}

def kmeans(myFood, parameter=6):
    """    
    Input:
        *myFood: A grid of two dimensional data with value True of False
        *parameter: The espected amount of foods in each cluster. Used to find out
        how many clusters are needed: k = len(food)/parameter
    
    Output:
        *new_centers(list): the k centers of the food. Each element in this list is 
        in the form of (tuple, int), where tuple indicating position and int is the 
        amount of foods in this cluster.
    
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

def getMissingFood(gmagent, gameState, steps=6):
    """
    Input:
        *gmagent: Game agent, as self in your agent function
        *gameState: the Game State
        *steps: the discounted factor, indicating how many moves ahead you are taking into account
        neglecting other previous histories
    
            
    Output:
        *ret_list: the return list, with the form of each element of (tuple, int), where tuple is the 
        position of the nearest food to the missing food and int is the steps away from current state.
        for example, (1,3),5, means the position (1,3) is the closest food to the missing food and it is 
        5 moves away from my current state
        
        
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
  

def CapsuleMonitor(gmagent, gameState, scare, last=40):
    """
    Input:
        gmagent: Game Agent, as self in your agent
        gameState: Game State
        scare: indicating if the agent is scared or not, initially 0, 1 if scared
        
    Output:
        scare(int): 0/1 indicating the current scare state 
    
    """
    
    if scare==0:
        index=-1
        preind=index-1
        if len(gmagent.observationHistory)>2:
            curCaplist=gmagent.getCapsulesYouAreDefending(gmagent.observationHistory[index])
            preCaplist=gmagent.getCapsulesYouAreDefending(gmagent.observationHistory[preind])
            if(len(preCaplist)-len(curCaplist)==1):
                return 1
    if scare==1 and len(gmagent.observationHistory)>2:
        if gameState.getAgentPosition(gmagent.index)==gmagent.observationHistory[0].getAgentPosition(gmagent.index):
            return 0
    return scare

def keyPositions(gmagent, gameState):
    """
    Input:
        gmagent: Game Agent
        gameState: Game State
        
    Output:
        validPositions(list):A list of key positions that is on or close(or 3 steps away from) the boundries
           
    
    """
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
    start_position=gmagent.observationHistory[0].getAgentPosition(gmagent.index)
    ##we dont want our key positions to be too close to the starting position
    validPositions=[pos for pos in validPositions if gmagent.getMazeDistance(pos,start_position)>=5]
    
    return validPositions

def aStarSearch(gmagent, gameState, goalPositions, startPosition=None, avoidPositions=[], returngoalPosition=False, returnCost= False):
    """
    Input:
        gmagent: Game Agent
        gameState: Game State
        goalPositions: A list, containing all the possible goal positions
        startPosition: The start position, by default it is the agent's current position        
        avoidPositions: A list containing all the opponents' ghost positions
        returngoalPosition: boolean, return the goal position if True
        
    Output:
        Path: A list containing the actions needed from current position to goal position
        CurrentPosition: return only when returngoalPosition is True. return the goal position
        
        
    """
    walls = gameState.getWalls()
    width = walls.width
    height = walls.height
    walls = walls.asList()
    if startPosition==None:
        startPosition=gmagent.getCurrentObservation().getAgentPosition(gmagent.index)
    
    actions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]
    actionVectors = [(int(Actions.directionToVector(action)[0]), int(Actions.directionToVector(action)[1])) for action in actions]
    currentPosition, currentPath, currentCost = startPosition, [], 0

    queue = util.PriorityQueueWithFunction(lambda entry: entry[2] +   # Total cost so far
                                           min(util.manhattanDistance(entry[0], endPosition) for endPosition in goalPositions))
                                           #width * height if entry[0] in avoidPositions else 0 +  # Avoid enemy locations like the plague
                                           
    # No Revisits
    visited = set([currentPosition])

    while currentPosition not in goalPositions:
        possiblePositions = [((currentPosition[0] + vector[0], currentPosition[1] + vector[1]), action) for vector, action in zip(actionVectors, actions)]
        legalPositions = [(position, action) for position, action in possiblePositions if position not in walls]
        for position, action in legalPositions:
            if position not in visited:
                visited.add(position)
                AvoidValue=0 
                if position in avoidPositions:
                    AvoidValue=width * height
                
                """value2=util.manhattanDistance(position, goalPositions[0])
                
                for endPosition in goalPositions:
                    if util.manhattanDistance(position, endPosition) <value2:
                        value2=util.manhattanDistance(position, endPosition)"""
                
                queue.push((position, currentPath + [action], currentCost + 1 + AvoidValue ))                
        if queue.isEmpty():##Just in case
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


def EscapePath(gmagent, gameState, returngoalPosition=False):
    ##astar:gmagent, gameState, goalPositions, startPosition=None, avoidPositions=[], returngoalPosition=False
    """
    Input:
        gmagent: Game Agent
        gameState: Current Game State
        returngoalPosition: not return position if it's False
        
    Output:
        A escape plan path to the boundries       
    
    IDEA:
        It can be implemented into Offensive Agent and used when the opponents' ghosts are
        within 2 steps away.
    """
    ##get ghost position
    ###get own position
    ####call A-star..
    myPos = gmagent.getCurrentObservation().getAgentPosition(gmagent.index)
    enemies = [gameState.getAgentState(i) for i in gmagent.getOpponents(gameState)]
    chaser = [a.getPosition() for a in enemies if not a.isPacman and a.getPosition() != None]
    
    walls = gameState.getWalls()
    height = walls.height
    walls = walls.asList()

    half_position=(int(gameState.data.layout.width/2-gmagent.red),int(gameState.data.layout.height/2))
    while(gameState.hasWall(half_position[0],half_position[1])):
        half_position=(half_position[0],half_position[1]-1) 
    
    goalPositions = [(half_position[0], height_position) for height_position in range(1, height-1) if not gameState.hasWall(half_position[0], height_position)]
    
    if len(chaser)!=0:
        return aStarSearch(gmagent, gameState, goalPositions=goalPositions, startPosition=myPos, avoidPositions=chaser, returngoalPosition=False)
    return []



def FindAlternativeFood(gmagent, gameState, returngoalPosition=True):
    myPos = gmagent.getCurrentObservation().getAgentPosition(gmagent.index)
    enemies = [gameState.getAgentState(i) for i in gmagent.getOpponents(gameState)]
    chasers=[]
    for a in enemies:
        if not a.isPacman:
            if a.getPosition()!=None:
                chasers.append(a.getPosition())
    
    #chasers = [a.getPosition() for a in enemies if  (not a.isPacman) and a.getPosition() != None]
    walls = gameState.getWalls()
    
    foodList = gmagent.getFood(gameState).asList()    
    
    height = walls.height
    width = walls.width
    walls = walls.asList()
    half_position=(int(gameState.data.layout.width/2-gmagent.red),int(gameState.data.layout.height/2))
    while(gameState.hasWall(half_position[0],half_position[1])):
        half_position=(half_position[0],half_position[1]-1) 
    
    
    goalPositions = foodList
    
    avoidPos=[]
    X=min(width/4, 3)
    Y=min(height/4, 3)
    
    for chaser in chasers:
        for posX in range(int(max(1,chaser[0]-X)), int(min(width,chaser[0]+X))):
            for posY in range(int(max(0,chaser[1]-Y)), int(min(height,chaser[1]+Y))):
                if not gameState.hasWall(posX, posY):
                    if (abs(posX-chaser[0])+abs(posY-chaser[1]))<=3:
                        avoidPos.append((posX, posY))
                    if (posX, posY) in goalPositions:
                        goalPositions.remove((posX,posY))
                        
    ##Here return a list and the position 
    currentPath, currentPosition=aStarSearch(gmagent, gameState, goalPositions=goalPositions, startPosition=myPos,
                                             avoidPositions=avoidPos, returngoalPosition=True)

    steps=min(5, len(currentPath))
    stackpath=[]
    if steps>0:
        for i in range(steps-1,-1,-1):
            stackpath.append(currentPath[i])
    return stackpath, currentPosition
       
 
last_seen_opponent=None

def CloggingOpponent(gmagent, gameState, returngoalPosition = False):
    """
    CLOGGING NOT EATING OPPONENTS!!!
    Input: 
        gmagent: Game Agent
        gameState: Game State
        
    Output: 
        True/False: Stop your agent when this function returns true.
    
    
    """
    myPos = gmagent.getCurrentObservation().getAgentPosition(gmagent.index)
    enemies = [gameState.getAgentState(i) for i in gmagent.getOpponents(gameState)]
    chaser = [a.getPosition() for a in enemies if a.isPacman and a.getPosition() != None]
    
   
    walls = gameState.getWalls()
    height = walls.height
    width = walls.width
    walls = walls.asList()
    #22 38 792
    half_position=(int(gameState.data.layout.width/2-gmagent.red),int(gameState.data.layout.height/2))
    while(gameState.hasWall(half_position[0],half_position[1])):
        half_position=(half_position[0],half_position[1]-1) 
        
    goalPositions = [(half_position[0], height_position) for height_position in range(3, height-1) if not gameState.hasWall(half_position[0], height_position)]
    avoidPositions=[myPos]
    
    startPosition=None
    
    if len(chaser)>0:
        startPosition=chaser[0]
        
    Path, Position, Cost =aStarSearch(gmagent, gameState, goalPositions, startPosition=startPosition, avoidPositions=avoidPositions, returngoalPosition=False, returnCost= True)
    
    if len(chaser)>1:
        Path2, Position2, Cost2 =aStarSearch(gmagent, gameState, goalPositions, startPosition=chaser[1], avoidPositions=avoidPositions, returngoalPosition=False, returnCost= True)
        if Cost2>Cost:
            Cost=Cost2
    
    if Cost > width * height and len(chaser)>0:
        return True
    
    #width * height
    if (len(gmagent.observationHistory)>=10):
        index_state=-2
        myPreState = gmagent.observationHistory[index_state]
        myPrePos = myPreState.getAgentPosition(gmagent.index)
        if (myPrePos!=myPos):
            return False
        while(myPrePos==myPos):
            index_state-=1
            myPreState = gmagent.observationHistory[index_state]
            myPrePos = myPreState.getAgentPosition(gmagent.index)
               
        index_state+=1
        myPreState = gmagent.observationHistory[index_state]
        myPrePos = myPreState.getAgentPosition(gmagent.index)
        
        Preenemies = [myPreState.getAgentState(i) for i in gmagent.getOpponents(myPreState)]
        Prechaser = [a.getPosition() for a in Preenemies if a.isPacman and a.getPosition() != None]
        
        #print("NOW===",myPrePos==myPos, myPrePos, myPos, index_state, len(Prechaser))
        while(index_state!=-2):
            myPreState = gmagent.observationHistory[index_state]
            myPrePos = myPreState.getAgentPosition(gmagent.index)
            Preenemies = [myPreState.getAgentState(i) for i in gmagent.getOpponents(myPreState)]
            Prechaser = [a.getPosition() for a in Preenemies if a.isPacman and a.getPosition() != None]
            if len(Prechaser) == 0:
                break
            index_state += 1
        index_state -= 1
        myPreState = gmagent.observationHistory[index_state]
        myPrePos = myPreState.getAgentPosition(gmagent.index)
        #print("AFTER===",myPrePos==myPos, myPrePos, myPos, index_state, len(Prechaser))
        Preenemies = [myPreState.getAgentState(i) for i in gmagent.getOpponents(myPreState)]
        Prechaser = [a.getPosition() for a in Preenemies if a.isPacman and a.getPosition() != None]
        if len(Prechaser)>0 and gmagent.getMazeDistance(myPrePos,Prechaser[0])>3: return True
    
    return False

def CheckIfAgentStucking(gmagent, gameState, referhistory=10, countingfactor=3):
    """
    !!!JUST FOR OFFENSIVE AGENT!!!
    return True when the agent is stuck
    
    """
    
    referhistory=min(referhistory,len(gmagent.observationHistory))
    curposition=gameState.getAgentPosition(gmagent.index)
    
    for i in range(-1, -1-referhistory, -1):
        historyposition2=gmagent.observationHistory[i].getAgentPosition(gmagent.index)
        if curposition==historyposition2:
            countingfactor-=1
    return countingfactor<0


def RunForestCheckDeadAlley(gmagent, gameState, action):
    """
    Call this function when you are in urgent running ignoring foods
    RETURN: True when this direction is dangerous
    """
    walls = gameState.getWalls()
    width = walls.width
    height = walls.height
    walls = walls.asList()
    startPosition=gmagent.getCurrentObservation().getAgentPosition(gmagent.index)
    avoidPos=[startPosition]
    
    half_position=(int(gameState.data.layout.width/2-gmagent.red),int(gameState.data.layout.height/2))
    while(gameState.hasWall(half_position[0],half_position[1])):
        half_position=(half_position[0],half_position[1]-1) 
        
    goalPositions = [(half_position[0], height_position) for height_position in range(3, height-1) if not gameState.hasWall(half_position[0], height_position)]
    
    successor = gmagent.getSuccessor(gameState, action)
        
    myState = successor.getAgentState(gmagent.index)
    successorPos = myState.getPosition()
    Path, Position, Cost =aStarSearch(gmagent, gameState, goalPositions, startPosition=successorPos, avoidPositions=avoidPos, returngoalPosition=False, returnCost= True)
    if Cost > width * height:
        return True
    #width * height
    return False




def LoopBreakerMoniter(gmagent, gameState, referhistory=12):
    """
    
    ####FOR OFFENSIVE AGENTS####    
    Return True when our agents need to break a loop..
    
    Logic: If our distance from the ghost remain the same for last 12(referhistroy) states
        Then we need to break the loop by CHANGING TAGET POSITION TO THE BOUNDRIES
    """
    if len(gmagent.observationHistory)<=referhistory: return False
    
    myPos = gmagent.getCurrentObservation().getAgentPosition(gmagent.index)
    enemies = [gameState.getAgentState(i) for i in gmagent.getOpponents(gameState)]
    chaser = [a.getPosition() for a in enemies if (not a.isPacman) and a.getPosition() != None]
    
    if len(chaser)>0:
        referDistance = gmagent.getMazeDistance(myPos, chaser[0])
    else:return False
    for i in range(-referhistory, -1):
        myPreState = gmagent.observationHistory[i]
        myPrePos = myPreState.getAgentPosition(gmagent.index)
        Preenemies = [myPreState.getAgentState(i) for i in gmagent.getOpponents(myPreState)]
        Prechaser = [a.getPosition() for a in Preenemies if (not a.isPacman) and a.getPosition() != None]
        if len(Prechaser)==0:
            return False
        PreDist = gmagent.getMazeDistance(myPrePos, Prechaser[0])
        if PreDist != referDistance:
            return False
    return True
  








