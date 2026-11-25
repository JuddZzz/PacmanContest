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
from collections import deque
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

        self.returning = False

    def choose_action(self, game_state):
        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()
        closest_danger_ghost = 10000000

        for ghost_i in self.get_opponents(game_state): #we look for every ghost
            this_ghost = game_state.get_agent_state(ghost_i) # this ghost
            ghost_pos = this_ghost.get_position() #position of this ghost

            if not this_ghost.is_pacman: #closest ghost
                if this_ghost.scared_timer <= 2:
                    if ghost_pos != None:
                        distance = self.get_maze_distance(my_pos, nearest_point(ghost_pos))
                        closest_danger_ghost = min(closest_danger_ghost,distance)

        if my_state.num_carrying == 0: #if i have no food, i don't go to the base
            self.returning = False
        else:
            moves_left = game_state.data.timeleft // 4 #moves left
            food_left = len(self.get_food(game_state).as_list())
            if (my_state.num_carrying >= 4 or closest_danger_ghost <= 4 or food_left <= 2 or moves_left <= 15): #for any of this cases, we go to the base
                self.returning = True

        actions = game_state.get_legal_actions(self.index) #my actions
        if Directions.STOP in actions and len(actions) > 1:
            actions.remove(Directions.STOP) #we don't want to stop
        action_values = []
        for action in actions:
            action_values.append(self.evaluate(game_state, action)) #evaluate action i

        best_value = max(action_values) #best value
        best_actions = [] #best actions
        for action_index in range(len(actions)):
            if action_values[action_index] == best_value:
                best_actions.append(actions[action_index]) #we get the best actions that are equal to the best value


        return random.choice(best_actions) #we return one of the best actions

        
    def base_distance(self, position, game_state):
        closest_danger_ghost = []

        for ghost_i in self.get_opponents(game_state): #we get the closest ghost
            this_ghost = game_state.get_agent_state(ghost_i)
            ghost_pos = this_ghost.get_position()            
            if not this_ghost.is_pacman:
                if this_ghost.scared_timer == 0:
                    if ghost_pos != None:
                        closest_danger_ghost.append(nearest_point(ghost_pos))

        best_min_distance = 100000000

        for home_i in self.home_positions: #best distance to base
            this_dist = self.get_maze_distance(position, home_i)
            cost = this_dist
            for ghost_i in closest_danger_ghost:
                distance = self.get_maze_distance(home_i, ghost_i)
                cost += max(0, 4 - distance)

            best_min_distance = min(best_min_distance, cost)

        return best_min_distance

    def _is_risky_dead_end(self, game_state, my_pos, target, dangerous_ghosts):
        if not dangerous_ghosts or target is None:
            return False

        walls = game_state.get_walls()
        my_pos = nearest_point(my_pos)
        target = nearest_point(target)

        # See how many possible exits we have if we go to this path
        x, y = target
        open_dirs = 0
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < walls.width and 0 <= ny < walls.height and not walls[nx][ny]:
                open_dirs += 1

        # In case this path is small and a ghost is nearby... 
        if open_dirs <= 2:
            my_dist = self.get_maze_distance(my_pos, target)
            for g_pos in dangerous_ghosts:
                g_dist = self.get_maze_distance(g_pos, target)
                # If the ghost blocks our exit, this path is risky
                if g_dist <= my_dist + 3:
                    return True
        return False
    
    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # self.getScore(successor)
        agent_state = successor.get_agent_state(self.index) #state of successor
        my_pos = agent_state.get_position() #position of successor

        closest_danger_ghost = 10000 #ghost not scared
        dangerous_ghosts = []
        for ghost_i in self.get_opponents(successor):
            this_ghost = successor.get_agent_state(ghost_i)
            ghost_pos = this_ghost.get_position()

            if not this_ghost.is_pacman: #dangerous ghost
                if this_ghost.scared_timer <= 2: #take into account this ghost if we want to eat a capsule
                    if ghost_pos != None:
                        ghost_pos = nearest_point(ghost_pos)
                        dangerous_ghosts.append(ghost_pos)
                        this_dist = self.get_maze_distance(my_pos, ghost_pos)
                        closest_danger_ghost = min(closest_danger_ghost, this_dist)

        if food_list:
            if agent_state.is_pacman: # if we are pacman, we search for food
                min_distance = float('inf')
                for food in food_list:
                    distance = self.get_maze_distance(my_pos, food)
                    if self._is_risky_dead_end(successor, my_pos, food, dangerous_ghosts): #risky path
                        distance += 12 #penalize this risky path
                    if distance < min_distance:
                        min_distance = distance

            else: #if we are a ghost, we search for the best entrance
                walls = successor.get_walls()
                if self.red:
                    entry_x = walls.width // 2
                else:
                    entry_x = walls.width // 2 - 1

                entry_positions = [] #possible entries to enemy base
                for y in range(walls.height):
                    if not walls[entry_x][y]:
                        entry_positions.append((entry_x, y))

                min_distance = float('inf')
                if entry_positions: 
                    for food in food_list: #food and entrance
                        for entry in entry_positions:
                            route_cost = self.get_maze_distance(my_pos, entry)
                            route_cost += self.get_maze_distance(entry, food)

                            if dangerous_ghosts: #danger ghost
                                nearest_defender = float('inf')
                                for ghost in dangerous_ghosts:
                                    distance = self.get_maze_distance(entry, ghost)
                                    if distance < nearest_defender:
                                        nearest_defender = distance

                                if nearest_defender <= 2:
                                    route_cost += 25
                                else:
                                    route_cost += max(0, 4 - nearest_defender)

                            if route_cost < min_distance:
                                min_distance = route_cost
                else:
                    for food in food_list:
                        distance = self.get_maze_distance(my_pos, food)
                        if distance < min_distance:
                            min_distance = distance

            features['distance_to_food'] = min_distance

        features['danger'] = (
                max(0, 5 - closest_danger_ghost) * (1 + agent_state.num_carrying)
            )
        
        if not agent_state.is_pacman and my_pos is not None: #we are in our base and there is an opponent nearby eating dots.
            for opp_i in self.get_opponents(successor):
                opp_state = successor.get_agent_state(opp_i)
                opp_pos = opp_state.get_position()
                if opp_state.is_pacman and opp_pos is not None:
                    opp_dist = self.get_maze_distance(my_pos, nearest_point(opp_pos))
                    if agent_state.scared_timer > 0 and opp_dist <= 3:
                        features['danger'] += (4 - opp_dist) * 5

        capsules = self.get_capsules(successor) #capsules to scare the ghost
        if capsules and closest_danger_ghost <= 6 and (not self.returning or agent_state.num_carrying < 4): #if we don't ho home and there is a dangerous ghost
            capsule_distance = float('inf')
            for capsule in capsules: #find a capsule
                distance = self.get_maze_distance(my_pos, capsule)
                if distance < capsule_distance:
                    capsule_distance = distance #get the closest capsule
            features['capsule_distance'] = capsule_distance 

        features['return_cost'] = ( #If we have a lot of food we would like to return to the base
            agent_state.num_carrying * self.base_distance(my_pos,game_state)
        )        
        
        return features

    def get_weights(self, game_state, action):
        if self.returning: #values if we want to go home
            return {'successor_score': 0, 'distance_to_food': 0, 'return_cost': -20, 'danger': -8, 'capsule_distance': -10,}
        #values if we are attacking
        return {'successor_score': 100, 'distance_to_food': -1, 'return_cost': -5, 'danger': -10, 'capsule_distance': -20,}


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state) # init the parent CaptureAgent

        walls = game_state.get_walls()  #determine the border column according to the team's side
        home_x = (walls.width // 2) - 1 if self.red else (walls.width // 2)

        #store all accessible positions along the border of two sides
        self.home_positions = [
            (home_x, y)
            for y in range(walls.height)
            if not walls[home_x][y]
        ]
        
        self.home_positions.sort(key=lambda pos: pos[1])

        #init the patrol route, moving from bottom to top
        self.patrol_index = 0
        self.patrol_direction = 1
        #keep track of defended food to detect recently invaded areas
        self.prev_food = self.get_food_you_are_defending(game_state).as_list()
        self.recent_loss_target = None

    def update_patrol_index(self, game_state):
        #avoid invalid indices
        if len(self.home_positions) <= 1:
            self.patrol_index = 0
            return

        my_pos = game_state.get_agent_position(self.index)
        if my_pos is None:
            return

        target = self.home_positions[self.patrol_index]

        #go to the next patrol position when the current one is reached
        if nearest_point(my_pos) == target:

            next_index = self.patrol_index + self.patrol_direction

            #reverse direction at the end of the patrol route
            if next_index >= len(self.home_positions):
                self.patrol_direction = -1
                next_index = len(self.home_positions) - 2
            elif next_index < 0:
                self.patrol_direction = 1
                next_index = 1
            self.patrol_index = max(0, min(len(self.home_positions) - 1, next_index))

    def _update_lost_food_target(self, game_state):  #compare current defended food with the previous observation
        current_food = self.get_food_you_are_defending(game_state).as_list()
        eaten = [pos for pos in self.prev_food if pos not in current_food]
        self.prev_food = current_food

        my_pos = game_state.get_agent_position(self.index)

        #if food has disappeared, investigate the closest affected position
        if eaten and my_pos is not None:
            self.recent_loss_target = min(eaten, key=lambda pos: self.get_maze_distance(nearest_point(my_pos), pos))

        #clear target once the agent reaches the suspected invasion area
        elif self.recent_loss_target is not None and my_pos is not None:
            if nearest_point(my_pos) == self.recent_loss_target:
                self.recent_loss_target = None

    def choose_action(self, game_state):
        self._update_lost_food_target(game_state)  #update the suspected invasion area before selecting an action

        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]  #identify opponents whose positions are currently visible
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]

        if not invaders: #continue the patrol route when no invader is visible
            self.update_patrol_index(game_state)

        actions = game_state.get_legal_actions(self.index)  #evaluate all legal actions and select one with the highest score
        
        #try to not stop moving when other movements are available
        if Directions.STOP in actions and len(actions) > 1:
            actions.remove(Directions.STOP)

        values = [self.evaluate(game_state, a) for a in actions]
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        return random.choice(best_actions)

    def _estimate_hidden_invader(self, game_state): #estimate where can be the attacker
        my_pos = game_state.get_agent_position(self.index)
        my_pos = nearest_point(my_pos)
        noisy = game_state.get_agent_distances()
        if noisy is None: #no current target
            return self.recent_loss_target

        walls = game_state.get_walls() #walls
        middle = walls.width // 2
        candidates = []

        for i in self.get_opponents(game_state): #pacman
            opp_state = game_state.get_agent_state(i)
            if not opp_state.is_pacman: #this is not a pacman
                continue
            opp_pos = opp_state.get_position() #otherwise, get his position
            if opp_pos is not None: #we see the pacman. We go for him
                return nearest_point(opp_pos)

            dist_est = noisy[i] #if we don't see the pacman but he is somewhere, we estimate his position
            for x in range(walls.width):
                for y in range(walls.height): #search
                    if walls[x][y]:
                        continue
                    on_our_half = (x < middle) if self.red else (x >= middle) #pacman must be on our half
                    if not on_our_half: #pacman not on our half
                        continue
                   
                    if util.manhattan_distance(my_pos, (x, y)) <= 5: #pacman at 5 steps away from us. we will see him so we do not need to estimate
                        continue
                    prob = game_state.get_distance_prob(
                        util.manhattan_distance(my_pos, (x, y)), dist_est #we estimate where the enemy is
                    )
                    if prob > 0:
                        candidates.append((prob, (x, y)))

        if candidates:
            candidates.sort(key=lambda item: item[0], reverse=True)
            return candidates[0][1] #most probable position of our opponent
        return self.recent_loss_target #no current target

    def get_features(self, game_state, action):
        features = util.Counter()
        #generate the successor state resulting from the proposed action
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()
        walls = game_state.get_walls()
        middle = walls.width // 2

        #reward remaining in our territory and penalize crossing the border so the defender doesn't leave its territory
        features['on_defense'] = 1 if not my_state.is_pacman else 0

        if my_pos is not None:
            next_x = nearest_point(my_pos)[0]
            if (self.red and next_x >= middle) or (not self.red and next_x < middle):
                features['crossed_border'] = 1

        #identify visible opponents currently invading our territory
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)

        if invaders and my_pos is not None:
            #prioritize invaders according to their distance and carried food
            self.recent_loss_target = None
            threats = []
            min_inv_dist = float('inf')
            for invader in invaders:
                inv_pos = invader.get_position()
                dist = self.get_maze_distance(nearest_point(my_pos), nearest_point(inv_pos))
                min_inv_dist = min(min_inv_dist, dist)
                carrying = invader.num_carrying
                threat = dist / (carrying + 1)
                threats.append(threat)

            features['invader_threat'] = min(threats)
            features['invader_distance'] = min_inv_dist

            #penalize actions that risk collision with an invader while scared
            if my_state.scared_timer > 0 and min_inv_dist <= 1:
                features['scared_collision_risk'] = 1

        else:
            #Patrol
            target = self.recent_loss_target

            #no visible target, but attacker somewhere
            any_invader_flag = any(a.is_pacman for a in enemies)
            if target is None and any_invader_flag:
                target = self._estimate_hidden_invader(successor)
               

            #if there is no suspected invasion area, continue patrolling 
            if target is None and self.home_positions:
                target = self.home_positions[self.patrol_index]

            current_pos = game_state.get_agent_position(self.index)

            #reward actions that reduce the distance to the current target so the agent follows the invader
            if target is not None and current_pos is not None and my_pos is not None:
                curr_dist = self.get_maze_distance(nearest_point(current_pos), target)
                next_dist = self.get_maze_distance(nearest_point(my_pos), target)
                features['patrol_progress'] = curr_dist - next_dist

        #discourage unnecessary stopping and reversing direction
        if action == Directions.STOP:
            features['stop'] = 1

        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        my_state = game_state.get_agent_state(self.index)
        
        #adapt the invader response according to the scared timer
        threat_weight = 6.0 if my_state.scared_timer > 0 else -12.0

        return {'num_invaders': -200,'on_defense': 150,'crossed_border': -1000,'invader_threat': threat_weight,'invader_distance': -2.0 if my_state.scared_timer == 0 else 0.0,'scared_collision_risk': -800,'patrol_progress': 8.0,'stop': -150,'reverse': -4.0,}