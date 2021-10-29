import time

from Env_Builder import *
from od_mstar3.col_set_addition import OutOfTimeError, NoSolutionError
from od_mstar3 import od_mstar
# from GroupLock import Lock
import random
from gym import spaces

'''
    Observation:
    Action space: (Tuple)
        agent_id: positive integer
        action: {0:STILL, 1:MOVE_NORTH, 2:MOVE_EAST, 3:MOVE_SOUTH, 4:MOVE_WEST,
                 5:NE, 6:SE, 7:SW, 8:NW, 5,6,7,8 not used in non-diagonal world}
    Reward: ACTION_COST for each action, GOAL_REWARD when robot arrives at target
'''


class Primal2Env(MAPFEnv):
    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self, observer, map_generator, num_agents=None,
                 IsDiagonal=False, frozen_steps=0, isOneShot=True):
        super(Primal2Env, self).__init__(observer=observer, map_generator=map_generator,
                                         num_agents=num_agents,
                                         IsDiagonal=IsDiagonal, frozen_steps=frozen_steps, isOneShot=isOneShot)

    def _reset(self, new_generator=None, *args):
        if new_generator is None:
            self.set_world()
        else:
            self.map_generator = new_generator
            self.world = World(self.map_generator, num_agents=self.num_agents, isDiagonal=self.IsDiagonal)
            self.num_agents = self.world.num_agents
            self.observer.set_env(self.world)

        self.fresh = True
        self.done = False
        self.done_asking = False

        if self.viewer is not None:
            self.viewer = None

    def step_all(self, movement_dict, msg_dict={}):
        """
        Agents are forced to freeze self.frozen_steps steps if they are standing on their goals.
        The new goal will be generated at the FIRST step it remains on its goal.

        :param movement_dict: {agentID_starting_from_1: action:int 0-4, ...}
                              unmentioned agent will be considered as taking standing still
        :return: obs_of_all:dict, reward_of_single_step:dict
        """

        # todo: freeze as PRIMAL1.
        # todo: Cancel the regenerating. one shot.

        # complete 'movement_dict'
        for agentID in range(1, self.num_agents + 1):
            if self.world.agents[agentID].freeze > self.frozen_steps:
                self.world.agents[agentID].freeze = 0   # set frozen agents free if enough steps
            if self.world.getDone(agentID) > 0 and self.isOneShot: # todo: fix the freeze function here
                movement_dict.update({agentID: 0})  # no action if Done
            if agentID not in movement_dict.keys() or self.world.agents[agentID].freeze:
                movement_dict.update({agentID: 0})  # add completed action list; add freezed ones.
            else:
                assert movement_dict[agentID] in list(range(5)) if self.IsDiagonal else list(range(9)), \
                    'action not in action space'
        # complete the 'msg_dict'
        for agentID in range(1, self.num_agents + 1):
            if agentID not in msg_dict.keys():
                msg_dict.update({agentID: 0.})

        if ENV_DEBUG_MODE:
            print('(step_all)movement_dict', movement_dict)
            print('(step_all)msg_dict', msg_dict)


        # detect the state after action
        status_dict, newPos_dict = self.world.CheckCollideStatus(movement_dict)
        if ENV_DEBUG_MODE:
            print('(step_all)status_dict', status_dict)
            print('(step_all)newPos_dict', newPos_dict)

        self.world.state[self.world.state > 0] = 0  # remove agents in the map for repose agents later
        put_goal_list = []
        freeze_list = []
        for agentID in range(1, self.num_agents + 1):
            # # whether env has done
            # if self.isOneShot and self.world.getDone(agentID) > 0:
            #     continue
            # self.done = False

            # pose agent at new position
            newPos = newPos_dict[agentID]
            # if self.isOneShot:
            #     if status_dict[agentID] not in [1, 2]:
            #         self.world.state[newPos] = agentID
            #     # else: don't place agents on state map
            # else:
            #     self.world.state[newPos] = agentID

            # if status_dict[agentID] not in [1, 2]:
            #     self.world.state[newPos] = agentID
            self.world.state[newPos] = agentID
            # agent state record
            self.world.agents[agentID].move(newPos, status_dict[agentID])

            # agent reward
            self.give_moving_reward(agentID)

            if status_dict[agentID] == 1:   # todo: check this in the future(is one shot? freezed?)
                if not self.isOneShot:
                    if self.world.agents[agentID].freeze == 0:
                        put_goal_list.append(agentID)
                    if self.world.agents[agentID].action_history[-1] == 0:  # standing still on goal
                        freeze_list.append(agentID)
                    self.world.agents[agentID].freeze += 1
                else:
                    if self.world.state[newPos] == 0:
                        self.world.state[newPos] = 0
                    self.world.agents[agentID].status = 2  # status=2 means done and removed from the env
                    self.world.goals_map[newPos] = 0

        # whether env is done
        self.done = True
        for agentID in range(1, self.num_agents + 1):
            # whether env has done
            if self.isOneShot and self.world.getDone(agentID) > 0:
                continue
            self.done = False

        # all of agents need to keep communicating with others
        # if put_goal_list and not self.isOneShot:
        #     self.world._put_goals(put_goal_list)
        #     for frozen_agent in freeze_list:
        #         free_agents.remove(frozen_agent)


        # Communication
        #   after action, update all the message on the communication channel at the new position.
        # Clear the communication channel
        self.world.reset_comms()
        # Update the communication channel
        for agentID in range(1, self.num_agents + 1):
            agent_pos = self.world.getPos(agentID)
            self.world.state_comms[agent_pos[0]][agent_pos[1]] = msg_dict[agentID]

        return self._observe(), self.individual_rewards


    def give_moving_reward(self, agentID):
        """
        WARNING: ONLY CALL THIS AFTER MOVING AGENTS!
        Only the moving agent that encounters the collision is penalized! Standing still agents
        never get punishment.
        """
        collision_status = self.world.agents[agentID].status
        if collision_status == 0:
            reward = self.ACTION_COST
            self.isStandingOnGoal[agentID] = False
        elif collision_status == 1:
            reward = self.ACTION_COST + self.GOAL_REWARD
            self.isStandingOnGoal[agentID] = True
            self.world.agents[agentID].dones += 1
        else:
            reward = self.ACTION_COST + self.COLLISION_REWARD
            self.isStandingOnGoal[agentID] = False

        self.individual_rewards[agentID] = reward

        # todo: check if this reward structure is same as the primal1


    def listValidActions(self, agent_ID, agent_obs):
        """
        :return: action:int, pos:(int,int)
        in non-corridor states:
            return all valid actions
        in corridor states:
            if standing on goal: Only going 'forward' allowed
            if not standing on goal: only going 'forward' allowed
        """

        def get_last_pos(agentID, position):
            """
            get the last different position of an agent
            """
            history_list = copy.deepcopy(self.world.agents[agentID].position_history)
            history_list.reverse()
            assert (history_list[0] == self.world.getPos(agentID))
            history_list.pop(0)
            if history_list == []:
                return None
            for pos in history_list:
                if pos != position:
                    return pos
            return None

        VANILLA_VALID_ACTIONS = True   

        if VANILLA_VALID_ACTIONS == True:                                       ##### set true here. what is VANILLA?
            available_actions = []
            pos = self.world.getPos(agent_ID)
            available_actions.append(0)  # standing still always allowed 
            num_actions = 4 + 1 if not self.IsDiagonal else 8 + 1
            for action in range(1, num_actions):                                # already append the standing above.
                direction = action2dir(action)
                new_pos = tuple_plus(direction, pos)
                lastpos = None
                try:
                    lastpos = self.world.agents[agent_ID].position_history[-2]
                except:
                    pass
                if new_pos == lastpos:
                    continue
                if self.world.state[new_pos[0], new_pos[1]] == 0:
                    available_actions.append(action)

            return available_actions

        available_actions = []
        pos = self.world.getPos(agent_ID)
        # if the agent is inside a corridor
        if self.world.corridor_map[pos[0], pos[1]][1] == 1:
            corridor_id = self.world.corridor_map[pos[0], pos[1]][0]
            if [pos[0], pos[1]] not in self.world.corridors[corridor_id]['StoppingPoints']:
                possible_moves = self.world.blank_env_valid_neighbor(*pos)
                last_position = get_last_pos(agent_ID, pos)
                for possible_position in possible_moves:
                    if possible_position is not None and possible_position != last_position \
                            and self.world.state[possible_position[0], possible_position[1]] == 0:
                        available_actions.append(dir2action(tuple_minus(possible_position, pos)))

                    elif len(self.world.corridors[corridor_id]['EndPoints']) == 1 and possible_position is not None \
                            and possible_moves.count(None) == 3:
                        available_actions.append(dir2action(tuple_minus(possible_position, pos)))

                if not available_actions:
                    available_actions.append(0)
            else:
                possible_moves = self.world.blank_env_valid_neighbor(*pos)
                last_position = get_last_pos(agent_ID, pos)
                if last_position in self.world.corridors[corridor_id]['Positions']:
                    available_actions.append(0)
                    for possible_position in possible_moves:
                        if possible_position is not None and possible_position != last_position \
                                and self.world.state[possible_position[0], possible_position[1]] == 0:
                            available_actions.append(dir2action(tuple_minus(possible_position, pos)))
                else:
                    for possible_position in possible_moves:
                        if possible_position is not None \
                                and self.world.state[possible_position[0], possible_position[1]] == 0:
                            available_actions.append(dir2action(tuple_minus(possible_position, pos)))
                    if not available_actions:
                        available_actions.append(0)
        else:
            available_actions.append(0)  # standing still always allowed 
            num_actions = 4 + 1 if not self.IsDiagonal else 8 + 1
            for action in range(1, num_actions):
                direction = action2dir(action)
                new_pos = tuple_plus(direction, pos)
                lastpos = None
                blocking_valid = self.get_blocking_validity(agent_obs, agent_ID, new_pos)
                if not blocking_valid:
                    continue
                try:
                    lastpos = self.world.agents[agent_ID].position_history[-2]
                except:
                    pass
                if new_pos == lastpos:
                    continue
                if self.world.corridor_map[new_pos[0], new_pos[1]][1] == 1:
                    valid = self.get_convention_validity(agent_obs, agent_ID, new_pos)
                    if not valid:
                        continue
                if self.world.state[new_pos[0], new_pos[1]] == 0:
                    available_actions.append(action)

        return available_actions

    def get_blocking_validity(self, observation, agent_ID, pos):
        top_left = (self.world.getPos(agent_ID)[0] - self.obs_size // 2,
                    self.world.getPos(agent_ID)[1] - self.obs_size // 2)
        blocking_map = observation[0][5]
        if blocking_map[pos[0] - top_left[0], pos[1] - top_left[1]] == 1:
            return 0
        return 1

    def get_convention_validity(self, observation, agent_ID, pos):
        top_left = (self.world.getPos(agent_ID)[0] - self.obs_size // 2,
                    self.world.getPos(agent_ID)[1] - self.obs_size // 2)
        blocking_map = observation[0][5]
        if blocking_map[pos[0] - top_left[0], pos[1] - top_left[1]] == -1:
            deltay_map = observation[0][7]
            if deltay_map[pos[0] - top_left[0], pos[1] - top_left[1]] > 0:
                return 1
            elif deltay_map[pos[0] - top_left[0], pos[1] - top_left[1]] == 0:
                deltax_map = observation[0][6]
                if deltax_map[pos[0] - top_left[0], pos[1] - top_left[1]] > 0:
                    return 1
                else:
                    return 0
            elif deltay_map[pos[0] - top_left[0], pos[1] - top_left[1]] < 0:
                return 0
            else:
                print('Weird')
        else:
            return 1




class DummyEnv(Primal2Env):
    def __init__(self, observer, map_generator, num_agents=None, IsDiagonal=False):
        super(DummyEnv, self).__init__(observer=observer, map_generator=map_generator,
                                       num_agents=num_agents,
                                       IsDiagonal=IsDiagonal)

    def _render(self, mode='human', close=False, screen_width=800, screen_height=800):
        pass


if __name__ == '__main__':
    from matplotlib import pyplot
    from Primal2Observer import Primal2Observer
    from Map_Generator2 import maze_generator # from Map_Generator import maze_generator
    from Map_Generator2 import manual_generator # from Map_Generator import manual_generator

    # state0 = [[-1, -1, -1, -1, -1, -1, -1],
    #           [-1, 1, -1, 0, 0, 0, -1],
    #           [-1, 0, -1, -1, -1, 0, -1],
    #           [-1, 0, 0, 0, -1, 0, -1],
    #           [-1, 0, -1, 0, 0, 0, -1],
    #           [-1, 2, -1, 0, 0, 0, -1],
    #           [-1, -1, -1, -1, -1, -1, -1]]
    n_agents = 8
    env = Primal2Env(num_agents=n_agents,
                     observer=Primal2Observer(observation_size=7),
                     map_generator=maze_generator(env_size=(10, 11),
                                                  wall_components=(5, 10), obstacle_density=(0.1, 0.2)),
                     IsDiagonal=False, isOneShot=True)

    env._reset()
    env._render()
    print('(PrimalC(__main__))env.world.state', env.world.state)
    print('(PrimalC(__main__))env.world.goals_map', env.world.goals_map)
    for j in range(0, 50):
         a1 = int(input())
         a2 = int(input())
         a3 = int(input())
         a4 = int(input())
         a5 = int(input())
         a6 = int(input())
         a7 = int(input())
         a8 = int(input())
         # a2 = a3 = a4 = a5 = a6 = a7 = a8 = a1
         movement = {1: a1, 2: a2, 3: a3, 4: a4, 5: a5, 6: a6, 7: a7, 8: a8}
         msg = {1: a1, 2: a2, 3: a3, 4: a4, 5: a5, 6: a6, 7: a7, 8: a8}
         env.step_all(movement_dict=movement, msg_dict=msg)
         obs = env._observe()
         env._render()
         print('(PrimalC(__main__))env.world.state', env.world.state)
         print('(PrimalC(__main__))env.world.goals_map', env.world.goals_map)
         print('(PrimalC(__main__))env.world.state_comms', env.world.state_comms)
         print('(PrimalC(__main__))obs[1]', obs[1])
