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

        if food_left <= 2:
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

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # self.getScore(successor)

        # Compute distance to the nearest food

        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -1}


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

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

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}



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

    def evaluation_function(self, current_game_state, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (new_food) and Pacman position after moving (new_pos).
        new_scared_times holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successor_game_state = current_game_state.generate_pacman_successor(action)  #next gamestate
        new_pos = successor_game_state.get_pacman_position() #Pacman new position
        new_food = successor_game_state.get_food() #remaining food
        new_ghost_states = successor_game_state.get_ghost_states() #states of each ghost (position of each ghost)
        new_scared_times = [ghostState.scared_timer for ghostState in new_ghost_states] #time remaining each ghost is scared


        
        "*** YOUR CODE HERE ***"
        food_list = new_food.as_list() #coords of each food

        if not food_list: #there is no food
            return 1000000 #we won

        food_distances = []
        for i in food_list:
            dist = util.manhattan_distance(new_pos,i) #we store the distance from our pacman to
            food_distances.append(dist)               #each food

        closest_food_dist = min(food_distances) #closest food



        closest_danger_ghost = 10000 #ghost not scared
        for ghostState in new_ghost_states:
            time_scared_ghost = ghostState.scared_timer #time remaining ghost i is scared
            pos_ghost = ghostState.get_position() #position of ghost i
            dist_ghost = util.manhattan_distance(new_pos,pos_ghost) #distance between this ghost and pacman

            if time_scared_ghost == 0: #if the ghost is aggressive
                if dist_ghost<closest_danger_ghost: ##if this distance is closer to pacman...
                    closest_danger_ghost = dist_ghost #this is the new dangerous ghost

        score = successor_game_state.get_score() #modify score after we see the next food and ghost

        if closest_danger_ghost<=2:
            score-=1000 #dangerous ghost, we must leave. we penalize more
        elif closest_danger_ghost<=5:
            score-=100 #ghost not too close, but he is approaching

        score+=1/closest_food_dist #the closer the better. 
                                   #if it is not very close, the division helps us penalize the score

        return score

def score_evaluation_function(current_game_state):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return current_game_state.get_score()


class AlphaBetaAgent(CaptureAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def get_action(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluation_function
        """
        "*** YOUR CODE HERE ***"
        def min_score(game_state,agent_index,depth,alpha,beta): #returns the minimum score

            v = float("inf") #minimum score
            poss_actions = game_state.get_legal_actions(agent_index) #actions that can do this ghost
            
            num_agents = game_state.get_num_agents() #number of agents
            next_agent = agent_index+1 #next agent

            #Base case
            if game_state.is_win():
                return self.evaluation_function(game_state) #Pacman win no matter the actions

            elif game_state.is_lose():
                return self.evaluation_function(game_state) #Pacman loses no matter the actions


            if next_agent >= num_agents: #if the next agent is the pacman (because we already did the actions of all ghosts), we reset next agent to 0
                next_agent = 0
                depth-=1 #Decrease depth because we finished 1 depth

            if depth<0:
                return self.evaluation_function(game_state) #end of the depth
            
            #Recursive case
            for i in poss_actions:
                next_game_state = game_state.generate_successor(agent_index,i) #next state

                if next_agent!=0: #if the next agent is not pacman
                    score = min_score(next_game_state,next_agent,depth,alpha,beta) #we receive the min score from the next state           


                else: #if the next agent is pacman
                    score = max_score(next_game_state,next_agent,depth,alpha,beta) #we receive the max score from the next state

                v = min(score,v) #we update v to the minimum between v and score

                if v<alpha: #alpha-cut: alpha is greater than v. We return v
                    return v

                if v<beta: #we update beta if v is smaller than beta
                    beta = v    

            return v #return v


        def max_score(game_state,agent_index,depth,alpha,beta): #returns the maximum score 

            v = -float("inf") #max score
            poss_actions = game_state.get_legal_actions(agent_index) #actions that pacman can do
            next_agent = agent_index+1 #next agent

            #Base case
            if game_state.is_win(): 
                return self.evaluation_function(game_state) #Pacman win no matter the actions
            
            elif game_state.is_lose(): 
                return self.evaluation_function(game_state) #Pacman loses no matter the actions
            
            elif depth<=0: 
                return self.evaluation_function(game_state) #end of the depth
            
            #Recursive case
            for i in poss_actions:
                next_game_state = game_state.generate_successor(agent_index,i) #next game state 
            
                score = min_score(next_game_state,next_agent,depth,alpha,beta) #set score value to minimum

                v = max(score,v) #v is the max value between v and score

                if v>beta: #beta cut --> v is greater than beta
                    return v

                if v>alpha: #in case v is bigger than alpha, modify alpha value to v
                    alpha = v


            return v #return v

        #initialize variables

        #v > beta to do beta cut
        alpha = -float("inf")
        beta = float("inf")

        start_index = 0 #pacman starts
        depth = self.depth #depth of tree
        actions = game_state.get_legal_actions(start_index) #actions of pacman
        best_action = actions[0] #initialize best action
        v = -float("inf") #initialize highest score

        for i in actions: #loop through actions
            next_game_state = game_state.generate_successor(start_index,i) #get next game state
            score = min_score(next_game_state,start_index+1,depth,alpha,beta) #set total score as the minimum

            if score>v: #updating v
                v = score
                best_action = i #we update best action

            if v>beta: #beta cut since v>beta. we return the best action
                return best_action

            if v>alpha: #in case v is bigger than alpha, modify alpha value to v
                alpha = v

        return best_action #return the best action


        util.raise_not_defined()