# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --red_opts and --blue_opts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """
    low_food = True

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2 and self.low_food:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
    low_food = False
    def register_initial_state(self, game_state): 
        super().register_initial_state(game_state)

        walls = game_state.get_walls() #reiceve walls
        if self.red:
            home_x = walls.width // 2 - 1 #if we are red, this is our home
        
        else:
            home_x = walls.width // 2 #if we are blue, this is our home

        self.home_positions = [] 

        for i in range(walls.height): #we look at the middle point where the two bases are separated
            if not walls[home_x][i]: #if in this coordinate there is no wall
                self.home_positions.append((home_x,i)) #we found a path that let us go home
        

    def base_distance(self, position,game_state): #we compute the shortest and safest distance from our position to the base
        
        best_min_distance = 100000000
        for i in self.home_positions:
            this_dist = self.get_maze_distance(position, i)
            best_min_distance = min(best_min_distance,this_dist)
        return best_min_distance
    
    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # self.getScore(successor)
        agent_state = successor.get_agent_state(self.index) #state of successor
        my_pos = agent_state.get_position() #position of successor

        # Compute distance to the nearest food

        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance


        closest_danger_ghost = 10000 #ghost not scared
        for ghost_i in self.get_opponents(successor):
            this_ghost = successor.get_agent_state(ghost_i)
            ghost_pos = this_ghost.get_position()

            if not this_ghost.is_pacman:
                if this_ghost.scared_timer == 0:
                    if ghost_pos != None:
                        this_dist = self.get_maze_distance(my_pos,ghost_pos)
                        closest_danger_ghost = min(closest_danger_ghost,this_dist)
            

        features['danger'] = (
                max(0, 5 - closest_danger_ghost) * (1 + agent_state.num_carrying)
            )
        features['return_cost'] = ( #If we have a lot of food we would like to return to the base
            agent_state.num_carrying * self.base_distance(my_pos,game_state)
        )        
        
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -1, 'return_cost': -5, 'danger': -10}


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    #Patrol position (with this the defender patrols near the entrances where a invader can enter your base)
    def register_initial_state(self, game_state):
        super().register_initial_state(game_state) #initial position

        walls = game_state.get_walls() #gets if there's a wall

        if self.red:
            home_x = walls.width // 2 - 1 #if we're red team --> left, also // is integer division
        else:
            home_x = walls.width // 2 #if we're blue team

        self.home_positions = [] #create empty list to store positions we can go to in the border

        for i in range(walls.height): #visits all vertical coordinates
            if not walls[home_x][i]: #true if position is not a wall
                self.home_positions.append((home_x, i)) #store in the list we did the accessible positions to patrol in

        #Chooses a central entrance as patrol target
        self.patrol_target = self.home_positions[
            len(self.home_positions) // 2
        ]

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        if len(invaders) == 0: #when there's no invaders in the base, start patrol
            features['patrol_distance'] = self.get_maze_distance(
                my_pos, self.patrol_target
            )

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'patrol_distance':-2, 'stop': -100, 'reverse': -2}



class ReflexAgent(CaptureAgent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def get_action(self, game_state):
        """
        You do not need to change this method, but you're welcome to.

        get_action chooses among the best options according to the evaluation function.

        Just like in the previous project, get_action takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legal_moves = game_state.get_legal_actions()

        # Choose one of the best actions
        scores = [self.evaluation_function(game_state, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = random.choice(best_indices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legal_moves[chosen_index]

"""
def food_heuristic(state, problem):

    #function that computes h_n, in other words, the heuristic estimation

    position, food_grid = state

    "*** YOUR CODE HERE ***"
    food_list = food_grid.as_list() #list with food

    h_n = len(food_list) #h(n)
    
    if h_n == 0: #goal aware. There is no food, so the heuristic estimation is 0
        return 0

    A = -1 #Initialization of the max distance of a food from the pacman
    pos1 = None # Position (currently none) of A
    for i in food_list: # Iterate through the list of food
        h_n = util.manhattan_distance(i,position) # Manhattan distance --> abs( xy1[0] - xy2[0] ) + abs( xy1[1] - xy2[1] )
        if h_n>A: #Update A so it gets the needed max value
            A = h_n
            pos1 = i # Update coordinates of A

            
    B = -1 #We initialize B: It will have the maximum distance from pos1 to the next food
    for i in food_list: #we iterate through the list of food
            h_n = util.manhattan_distance(i,pos1) # Manhattan distance --> abs( xy1[0] - xy2[0] ) + abs( xy1[1] - xy2[1] )
            if h_n>B: #If this new distance is bigger than the current distance....
                B = h_n #We update B to the new distance

    return A+B #We return the sum of the 2 distances

"""

"""
def a_star_search(problem, heuristic=null_heuristic):
    "*** YOUR CODE HERE ***"

    visited = set() #list of visited paths. In the begginning, this is empty
    
    priorqueue = util.PriorityQueue() #We need a priority queue to know which path do we take  

    start_heuristic = heuristic(problem.get_start_state(),problem) #we initialize the heuristic funcion giving 2 arguments:
                                                                   #our problem
                                                                   #the initial position
    
    priorqueue.push((problem.get_start_state(),[],0),start_heuristic) #we store our initial position, 
    #the action we take (initially no action), initial cost (0), and the heuristic function to find the shortest distance
    
    while not priorqueue.is_empty():
        coords, direction, cost = priorqueue.pop() #we receive the state with less cost h(n) + g(n) (coordinates (x,y) of the possible visit) 
                                        #the type of movement (if we go north, south, east or west) 
                                        #and cost

        #case number 1: we reach the goal:
        if problem.is_goal_state(coords): 
            return direction #we return to which direction we went

        #case number 2: we didn't reach the goal, we need to find it

        if coords not in visited: #we only do the next lines of code if the node was not visited before
            visited.add(coords) #we store the coordinates because now they will be visited (so we won't visit it again)

            for i in problem.get_successors(coords): #i is formed of [(coordinates x,y),movement,cost]
                next_coords = i[0] #next coords (x,y)
                next_dir = i[1] #south,west...
                next_cost = i[2] #next Cost

                g_n = cost + next_cost #current total cost
                h_n = heuristic(next_coords,problem) #heuristic estimation
                total_sum = g_n + h_n #we do the sum of the total cost and the 
                #Manhattan distance heuristic

                if next_coords not in visited:  #We will do this movement because the coordinates were not visited
                    priorqueue.push((next_coords, direction + [next_dir],g_n),total_sum) #we store it for later

    util.raise_not_defined()
"""