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
from util import nearestPoint

random.seed(42)
#################
# Team creation #
#################


def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveMCT', second='TestDefensiveReflexAgent'):
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
                dist = self.getMazeDistance(self.start, pos2)
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


class TestOffensiveReflexAgent(TestReflexCaptureAgent):
    """
    A reflex agent that seeks food. This is an agent
    we give you to get an idea of what an offensive agent might look like,
    but it is by no means the best or only way to build an offensive agent.
    """

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        foodList = self.getFood(successor).asList()
        features['successorScore'] = -len(foodList)  # self.getScore(successor)

        # Compute distance to the nearest food

        if len(foodList) > 0:  # This should always be True,  but better safe than sorry
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food)
                               for food in foodList])
            features['distanceToFood'] = minDistance
        return features

    def getWeights(self, gameState, action):
        return {'successorScore': 100, 'distanceToFood': -1}

    def getRemainingScareTime(self,gameState, agentIndex):
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
        if myState.isPacman:
            features['onDefense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i)
                   for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition()
                    != None]
        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.getMazeDistance(
                myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        missingfoodinf = getMissingFood(self, gameState)
        # print(missingfoodinf)
        dist_miss = 0
        features['MissingFood'] = 0
        if len(missingfoodinf) > 0:
          # Weight should be modified
          features['MissingFood'] = 5
          # Try to use distance to measure what action should be taken
          for pos, i in missingfoodinf:
            dist_miss = self.getMazeDistance(pos, myPos)
        # print(5*dist_miss)
        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if myState.isPacman: features['onDefense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i)
                                           for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition()
                                                                       != None]
        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.getMazeDistance(
                myPos, a.getPosition()) for a in invaders]
            # print dists
            features['invaderDistance'] = min(dists)

        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(
            self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        return features

    def getWeights(self, gameState, action):
        return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}


# experimental test with MCT
class MCTBasedAgent(CaptureAgent):
    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        # print "Features",features
        # print "Weights",weights
        # print "Eval",features * weights
        return features * weights

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)
        return features

    def getWeights(self, gameState, action):
        return {'successorScore': 1.0}


class OffensiveMCT(MCTBasedAgent):

    def getFeatures(self, gameState, action):
        """
        Get features used for state evaluation.
        """
        features = util.Counter()

        successor = self.getSuccessor(gameState, action)
        # Compute score from successor state
        features['successorScore'] = self.getScore(gameState)
        
        if self.food_carrying >= self.MAX_FOOD_CARRYING or len(self.getFood(gameState).asList()) <= 2:
            features['successorScore'] = self.getScore(gameState)
        else:
            features['successorScore'] = -len(self.getFood(gameState).asList())

        foodList = self.getFood(successor).asList()
        """
        # compute distance to the nearest cluster of food : Not working properly, Pacman is getting stucked in some positions
        if len(foodList) > 0 and self.food_carrying <self.MAX_FOOD_CARRYING:
          successor_food_clusters=kmeans(self.getFood(successor),3)
          best_food_cluster=max(successor_food_clusters,
                                key=lambda item:item[1])[0]
          myPos = successor.getAgentState(self.index).getPosition()
          distance_to_food_cluster = self.getMazeDistance(
              myPos, best_food_cluster)
          print "Distance:",myPos,"-",best_food_cluster,"=>",distance_to_food_cluster
          features['distanceToFoodCluster'] = distance_to_food_cluster
        """

        # Compute distance to the nearest food. If it's in mode safe it shouldnt look for food.
        if len(foodList) > 0 and self.food_carrying < self.MAX_FOOD_CARRYING:
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food)
                               for food in foodList])
            features['distanceToFood'] = minDistance

        # Compute distance to closest ghost
        myPos = successor.getAgentState(self.index).getPosition()
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
                features['distanceToGhost'] = closestDist
                self.scared += 1
            else:
                self.scared = 0

        # featured in danger

        features['inDanger'] = 1 if self.scared else 0
        # Compute if is pacman
        features['isPacman'] = 1 if successor.getAgentState(
            self.index).isPacman else 0

        return features

    def getWeights(self, gameState, action):
        """
        Get weights for the features used in the evaluation.
        """

        successor = self.getSuccessor(gameState, action)
        myPos = successor.getAgentState(self.index).getPosition()
        enemies = [successor.getAgentState(i)
                   for i in self.getOpponents(successor)]
        inRange = filter(
            lambda x: not x.isPacman and x.getPosition() != None, enemies)
        if len(inRange) > 0:
            positions = [agent.getPosition() for agent in inRange]
            closestPos = min(
                positions, key=lambda x: self.getMazeDistance(myPos, x))
            closestDist = self.getMazeDistance(myPos, closestPos)
            closest_enemies = filter(
                lambda x: x[0] == closestPos, zip(positions, inRange))
            for agent in closest_enemies:
                if agent[1].scaredTimer > 0:
                    return {'successorScore': 200, 'distanceToFood': -5, 'distanceToGhost': 0, 'isPacman': 0}

        # return {'successorScore': 200, 'distanceToFood': -5, 'distanceToGhost': 2, 'isPacman': 0}
        return {'successorScore': 200, 'distanceToFood': -5, 'distanceToGhost': 10, 'isPacman': 0, 'inDanger': 5}
        # return {'successorScore': 200,'distanceToFood': -5, 'distanceToFoodCluster':-2, 'distanceToGhost': 2, 'isPacman': 0}

    def randomSimulation(self, depth, gameState):
        """
        Random simulate some actions for the agent. The actions other agents can take
        are ignored, or, in other words, we consider their actions is always STOP.
        The final state from the simulation is evaluated.
        """
        new_state = gameState.deepCopy()
        while depth > 0:
            # Get valid actions
            actions = new_state.getLegalActions(self.index)
            # The agent should not stop in the simulation
            actions.remove(Directions.STOP)
            current_direction = new_state.getAgentState(
                self.index).configuration.direction
            # get the reversed direction for the current direction
            reversed_direction = Directions.REVERSE[current_direction]
            # if is not the only move possible then discard it
            if reversed_direction in actions and len(actions) > 1:
                actions.remove(reversed_direction)
            # Randomly chooses a valid action
            a = random.choice(actions)
            # Compute new state and update depth
            new_state = new_state.generateSuccessor(self.index, a)
            depth -= 1
        # Evaluate the final simulation state
        return self.evaluate(new_state, Directions.STOP)

    def __init__(self, index):
        CaptureAgent.__init__(self, index)
        self.MAX_ITERATIONS = 30
        self.MAX_EXPLORATION = 15
        self.MAX_FOOD_CARRYING = 3
        self.scared = 0
        # Variables used to verify if the agent is locked

    # (15s max).
    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        # calculate all the distances
        self.distancer.getMazeDistances()

    #  (1s max).
    def chooseAction(self, gameState):
        # You can profile your evaluation time by uncommenting these lines
        start = time.time()

        # Get valid actions. Staying put is almost never a good choice, so
        # the agent will ignore this action.
        all_actions = gameState.getLegalActions(self.index)
        all_actions.remove(Directions.STOP)

        self.food_carrying = gameState.getAgentState(self.index).numCarrying

        q_a_s = {}
        for a in all_actions:
            new_state = gameState.generateSuccessor(self.index, a)
            sim_value = 0
            for iteration in range(self.MAX_ITERATIONS):
                sim_value += self.randomSimulation(
                    self.MAX_EXPLORATION, new_state)
            q_a_s[a] = sim_value

        # print "ACTIONS:",self.index,q_a_s
        next_play = max(q_a_s, key=q_a_s.get)

        # print 'eval time for offensive agent %d: %.4f, Action:%s' % (self.index, time.time() - start,next_play)
        return next_play

def writeToFile(self, gameState):
    food = self.getFood(gameState).asList()
    opponentFood = self.getFoodYouAreDefending(gameState).asList()
    score = self.getScore(gameState)
    capsules = self.getCapsules(gameState)
    capsulesOpponent = self.getCapsulesYouAreDefending(gameState)
    #print "Food:", len(food), "Defending Food :",len(opponentFood),"Score :",score


# experimantal test with UCT
class UCTBasedAgent(CaptureAgent):
    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        # print "Features",features
        # print "Weights",weights
        # print "Eval",features * weights
        return features * weights

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)
        return features

    def getWeights(self, gameState, action):
        return {'successorScore': 1.0}


class OffensiveUCT(UCTBasedAgent):

    def getFeatures(self, gameState, action):
        """
        Get features used for state evaluation.
        """
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        # Compute score from successor state
        features['successorScore'] = self.getScore(gameState)

        if self.food_carrying >= self.MAX_FOOD_CARRYING or len(self.getFood(gameState).asList()) <= 2:
            features['successorScore'] = self.getScore(gameState)
        else:
            features['successorScore'] = -len(self.getFood(gameState).asList())

        foodList = self.getFood(successor).asList()
        """
    # compute distance to the nearest cluster of food : Not working properly, Pacman is getting stucked in some positions
    if len(foodList) > 0 and self.food_carrying <self.MAX_FOOD_CARRYING:
      successor_food_clusters=kmeans(self.getFood(successor),3)
      best_food_cluster=max(successor_food_clusters,key=lambda item:item[1])[0]
      myPos = successor.getAgentState(self.index).getPosition()
      distance_to_food_cluster = self.getMazeDistance(myPos, best_food_cluster)
      print "Distance:",myPos,"-",best_food_cluster,"=>",distance_to_food_cluster
      features['distanceToFoodCluster'] = distance_to_food_cluster
    """

        # Compute distance to the nearest food. If it's in mode safe it shouldnt look for food.
        if len(foodList) > 0 and self.food_carrying < self.MAX_FOOD_CARRYING:
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food)
                               for food in foodList])
            features['distanceToFood'] = minDistance

        # Compute distance to closest ghost
        myPos = successor.getAgentState(self.index).getPosition()
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
                features['distanceToGhost'] = closestDist

        # Compute if is pacman
        features['isPacman'] = 1 if successor.getAgentState(
            self.index).isPacman else 0

        return features

    def getWeights(self, gameState, action):
        """
        Get weights for the features used in the evaluation.
        """

        successor = self.getSuccessor(gameState, action)
        myPos = successor.getAgentState(self.index).getPosition()
        enemies = [successor.getAgentState(i)
                   for i in self.getOpponents(successor)]
        inRange = filter(
            lambda x: not x.isPacman and x.getPosition() != None, enemies)
        if len(inRange) > 0:
            positions = [agent.getPosition() for agent in inRange]
            closestPos = min(
                positions, key=lambda x: self.getMazeDistance(myPos, x))
            closestDist = self.getMazeDistance(myPos, closestPos)
            closest_enemies = filter(
                lambda x: x[0] == closestPos, zip(positions, inRange))
            for agent in closest_enemies:
                if agent[1].scaredTimer > 0:
                    return {'successorScore': 200, 'distanceToFood': -5, 'distanceToGhost': 0, 'isPacman': 0}

        # return {'successorScore': 200, 'distanceToFood': -5, 'distanceToGhost': 2, 'isPacman': 0}
        return {'successorScore': 200, 'distanceToFood': -5, 'distanceToGhost': 2, 'isPacman': 0}
        # return {'successorScore': 200,'distanceToFood': -5, 'distanceToFoodCluster':-2, 'distanceToGhost': 2, 'isPacman': 0}

    def randomSimulation(self, depth, gameState):
        """
        Random simulate some actions for the agent. The actions other agents can take
        are ignored, or, in other words, we consider their actions is always STOP.
        The final state from the simulation is evaluated.
        """
        new_state = gameState.deepCopy()
        while depth > 0:
            # Get valid actions
            self.visited[new_state] += 1
            actions = new_state.getLegalActions(self.index)
            # The agent should not stop in the simulation
            actions.remove(Directions.STOP)
            current_direction = new_state.getAgentState(
                self.index).configuration.direction
            # get the reversed direction for the current direction
            reversed_direction = Directions.REVERSE[current_direction]
            # if is not the only move possible then discard it
            if reversed_direction in actions and len(actions) > 1:
                actions.remove(reversed_direction)
            # Randomly chooses a valid action
            a = random.choice(actions)
            # Compute new state and update depth
            new_state = new_state.generateSuccessor(self.index, a)
            depth -= 1

        # Evaluate the final simulation state
        # reward should be populated for each node
        self.rewards[new_state] = self.evaluate(new_state, Directions.STOP)
        return self.evaluate(new_state, Directions.STOP)

    def __init__(self, index):
        CaptureAgent.__init__(self, index)
        self.MAX_ITERATIONS = 30
        self.MAX_EXPLORATION = 15
        self.MAX_FOOD_CARRYING = 3
        self.visited = {}
        self.rewards = {}
        # Variables used to verify if the agent is locked

    # (15s max).
    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        # calculate all the distances
        self.distancer.getMazeDistances()

    #  (1s max).
    def chooseAction(self, gameState):
        # You can profile your evaluation time by uncommenting these lines
        start = time.time()

        # Get valid actions. Staying put is almost never a good choice, so
        # the agent will ignore this action.
        all_actions = gameState.getLegalActions(self.index)
        all_actions.remove(Directions.STOP)

        self.food_carrying = gameState.getAgentState(self.index).numCarrying

        q_a_s = {}
        for a in all_actions:
            new_state = gameState.generateSuccessor(self.index, a)
            sim_value = 0
            for iteration in range(self.MAX_ITERATIONS):
                sim_value += self.randomSimulation(
                    self.MAX_EXPLORATION, new_state)
            q_a_s[a] = sim_value

        # print "ACTIONS:",self.index,q_a_s
        next_play = max(q_a_s, key=q_a_s.get)

        # print 'eval time for offensive agent %d: %.4f, Action:%s' % (self.index, time.time() - start,next_play)
        return next_play


def kmeans(myFood, parameter=6):
    """
    myFood is grid variable defined in capture
       parameter is used to determine how many foods needed for a center.
       amount of food / parameter = round down to k
       e.g  20 foods with parameter=6 gives 3 centers(round down to 3)
            20 foods with parameter=5 gives 4 centers
    """

    # print myFood
    width = myFood.width
    height = myFood.height
    foodlist = [(i, j) for i in range(width)
                for j in range(height) if myFood[i][j] == True]
    k = max(1, len(foodlist)/parameter)

    if len(foodlist) > 0:
        centers_ = random.sample(foodlist, k)
        centers = [(i, 1) for i in centers_]

        while(1):
            new_clusters = [[i[0]] for i in centers]
            new_centers = []

            for i in foodlist:
                distance = distanceCalculator.manhattanDistance(
                    i, centers[0][0])
                index = 0
                for j in range(1, len(centers)):
                    dis = distanceCalculator.manhattanDistance(
                        i, centers[j][0])
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

                new_center = (
                    x_leng/len(new_clusters[i]), y_leng/len(new_clusters[i]))
                dis_close = 99999
                close_food = new_clusters[i][0]
                for j in range(len(new_clusters[i])):
                    dis2 = distanceCalculator.manhattanDistance(
                        new_clusters[i][j], new_center)
                    if dis2 < dis_close:
                        dis_close = dis2
                        close_food = new_clusters[i][j]
                new_centers.append((close_food, len(new_clusters[i])))
            if (new_centers == centers):
                break
            centers = new_centers
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
        # print(curfoodlist)
        # print(prefoodlist)
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
