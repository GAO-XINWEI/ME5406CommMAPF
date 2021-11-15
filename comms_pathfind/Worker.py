
import scipy.signal as signal
import copy
import numpy as np
import ray
import os
import imageio
from Env_Builder import *

from Map_Generator2 import maze_generator

from parameters import *

# helper functions
def discount(x, gamma):
    return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

class Worker():
    def __init__(self, metaAgentID, workerID, workers_per_metaAgent, env, localNetwork, sess, groupLock, learningAgent, global_step):

        self.metaAgentID = metaAgentID
        self.agentID = workerID
        self.name = "worker_" + str(workerID)
        self.num_workers = workers_per_metaAgent
        self.global_step = global_step
        self.nextGIF = 0

        self.env = env
        self.local_AC = localNetwork
        self.groupLock = groupLock
        self.learningAgent = learningAgent
        self.sess = sess
        self.loss_metrics = None
        self.perf_metrics = None
        self.allGradients = []

    def __del__(self):
        print('((worker)__del__)meta{0}worker{1}'.format(self.metaAgentID,self.agentID))

    def calculateImitationGradient(self, rollout, episode_count): # todo: check rollout
        rollout = np.array(rollout, dtype=object)
        # we calculate the loss differently for imitation
        # if imitation=True the rollout is assumed to have different dimensions:
        # [o[0],o[1],optimal_actions]
        target_meangoal = rollout[:, 2]
        target_block = rollout[:, 6]
        rewards = rollout[:, 7]
        advantages = rollout[:, 8]
        # rnn_state = self.local_AC.state_init
        # s1Value = self.sess.run(self.local_AC.value,
        #                         feed_dict={self.local_AC.inputs     : np.stack(rollout[:, 0]),
        #                                    self.local_AC.goal_pos   : np.stack(rollout[:, 1]),
        #                                    self.local_AC.state_in[0]: rnn_state[0],
        #                                    self.local_AC.state_in[1]: rnn_state[1]})[0, 0]
        #
        # v = self.sess.run([self.local_AC.value,
        #                    ],
        #                   # todo: feed the message(last time step) here
        #                   feed_dict={self.local_AC.inputs: np.stack(rollout[:, 0]),  # state
        #                   self.local_AC.goal_pos: np.stack(rollout[:, 1]),  # goal vector
        #                   self.local_AC.state_in[0]: rnn_state[0],
        #                   self.local_AC.state_in[1]: rnn_state[1],
        #                              })
        # values = v[0,0]
        # self.rewards_plus = np.asarray(rewards.tolist() + [s1Value])
        # discounted_rewards = discount(self.rewards_plus, gamma)[:-1]
        # self.value_plus = np.asarray(values.tolist() + [s1Value])
        # advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        # advantages = discount(advantages, gamma)

        temp_actions = np.stack(rollout[:, 3])
        rnn_state = self.local_AC.state_init
        feed_dict = {self.global_step             : episode_count,
                     self.local_AC.inputs         : np.stack(rollout[:, 0]),
                     self.local_AC.goal_pos       : np.stack(rollout[:, 1]),
                     self.local_AC.optimal_actions: np.stack(rollout[:, 3]),
                     self.local_AC.state_in[0]    : rnn_state[0],
                     self.local_AC.state_in[1]    : rnn_state[1],
                     self.local_AC.train_imitation: (rollout[:, 4]),
                     self.local_AC.target_v       : np.stack(temp_actions),
                     self.local_AC.train_value    : temp_actions,
                     # self.local_AC.advantages     : advantages,
                     self.local_AC.target_meangoals      : np.stack(target_meangoal),
                     self.local_AC.target_blockings      : np.stack(target_block),
                     }
        # print('feed ', feed_dict)

        v_l, i_l, local_vars, i_grads = self.sess.run([self.local_AC.value_loss,
                                           self.local_AC.imitation_loss,
                                           self.local_AC.local_vars,
                                           self.local_AC.i_grads
                                           ],
                                          feed_dict=feed_dict)
        if NN_DEBUG_MODE:
            print('v_l', v_l)
            print('i_l', i_l)
            # print('local_vars', local_vars)
            print('l_v', local_vars)
            # print('igrads', i_grads)

            # raise(TypeError)
        return [i_l], i_grads


    def calculateGradient(self, rollout, bootstrap_value, episode_count, rnn_state0):
        # ([s,a,r,s1,v[0,0]])

        rollout = np.array(rollout, dtype=object) # todo: meangoal, blocking
        inputs = rollout[:, 0]
        goals = rollout[:, 6]
        target_meangoal = rollout[:, 7]
        target_block = rollout[:, 8]
        # meangoal = rollout[:, -5]
        # blocking = rollout[:, -4]
        # message = rollout[:, -3]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        values = rollout[:, 4]
        valids = rollout[:, 5]
        train_value = rollout[:, -2]
        train_policy = rollout[:, -1]

        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns. (With bootstrapping)
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus, gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages, gamma)

        num_samples = min(EPISODE_SAMPLES, len(advantages))
        sampleInd = np.sort(np.random.choice(advantages.shape[0], size=(num_samples,), replace=False))

        feed_dict = {
            self.global_step         : episode_count,
            self.local_AC.target_v   : np.stack(discounted_rewards),
            self.local_AC.inputs     : np.stack(inputs),
            self.local_AC.goal_pos   : np.stack(goals),
            self.local_AC.actions    : actions,
            self.local_AC.target_meangoals  : np.stack(target_meangoal),
            self.local_AC.target_blockings  : np.stack(target_block),
            # self.local_AC.block      : block,
            # self.local_AC.message    : message,
            self.local_AC.train_valid: np.stack(valids),
            self.local_AC.advantages : advantages,
            self.local_AC.train_value: train_value,
            self.local_AC.state_in[0]: rnn_state0[0],
            self.local_AC.state_in[1]: rnn_state0[1],
            # self.local_AC.train_policy: train_policy,
            self.local_AC.train_valids: np.vstack(train_policy)
        }

        v_l, p_l, valid_l, e_l, g_n, v_n, blocking_l, meangoal_l, message_l, grads = self.sess.run([self.local_AC.value_loss,
                                                                self.local_AC.policy_loss,
                                                                self.local_AC.valid_loss,
                                                                self.local_AC.entropy,
                                                                self.local_AC.grad_norms,
                                                                self.local_AC.var_norms,
                                                                self.local_AC.blocking_loss,
                                                                self.local_AC.mean_goal_loss,
                                                                self.local_AC.message_loss,
                                                                self.local_AC.grads],
                                                                feed_dict=feed_dict)

        return [v_l, p_l, valid_l, e_l, blocking_l, meangoal_l, message_l, g_n, v_n], grads



    def imitation_learning_only(self, episode_count):
        self.env._reset()
        rollouts, targets_done = self.parse_path(episode_count)
        # rollouts.append([])
        if rollouts is None:
            return None, 0

        gradients = []
        losses = []
        for i in range(self.num_workers):
            train_buffer = rollouts[i]

            imitation_loss, grads = self.calculateImitationGradient(train_buffer, episode_count)

            gradients.append(grads)
            losses.append(imitation_loss)

        return gradients, losses



    def run_episode_multithreaded(self, episode_count, coord):
        if NN_DEBUG_MODE:
            print('(Worker-RL)Begin to run! meta:{0}, worker{1}'.format(self.metaAgentID, self.agentID))

        if self.metaAgentID < NUM_IL_META_AGENTS:
            assert(1==0)
            # print("THIS CODE SHOULD NOT TRIGGER")
            # self.is_imitation = True
            # self.imitation_learning_only()

        global episode_lengths, episode_mean_values, episode_invalid_ops, episode_stop_ops, episode_rewards, episode_finishes
        # print('episode_mean_values', episode_lengths)

        num_agents = self.num_workers

        with self.sess.as_default(), self.sess.graph.as_default():
            while self.shouldRun(coord, episode_count):
                episode_buffer, episode_values = [], []
                episode_reward = episode_step_count = episode_inv_count = targets_done =episode_stop_count = 0

                self.synchronize()

                # Initial state from the environment
                if self.agentID == 1:
                    if NN_DEBUG_MODE:
                        print('(Worker-RL)self.env._reset(a) meta:{0}, worker{1}'.format(self.metaAgentID, self.agentID))
                    self.env._reset()
                    if NN_DEBUG_MODE:
                        print('(Worker-RL)self.env._reset(b) meta:{0}, worker{1}'.format(self.metaAgentID, self.agentID))
                    joint_observations[self.metaAgentID] = self.env._observe()

                if NN_DEBUG_MODE:
                    print('(Worker-RL)self.synchronize(1a) meta:{0}, worker{1}'.format(self.metaAgentID, self.agentID))
                self.synchronize()  # synchronize starting time of the threads
                if NN_DEBUG_MODE:
                    print('(Worker-RL)self.synchronize(1b) meta:{0}, worker{1}'.format(self.metaAgentID, self.agentID))

                # Get Information For Each Agent
                validActions = self.env.listValidActions(self.agentID, joint_observations[self.metaAgentID][self.agentID])

                s = joint_observations[self.metaAgentID][self.agentID]

                rnn_state = self.local_AC.state_init
                rnn_state0 = rnn_state


                self.synchronize()  # synchronize starting time of the threads
                swarm_reward[self.metaAgentID] = 0
                swarm_targets[self.metaAgentID] = 0

                episode_rewards[self.metaAgentID] = []
                episode_finishes[self.metaAgentID] = []
                episode_lengths[self.metaAgentID] = []
                episode_mean_values[self.metaAgentID] = []
                episode_invalid_ops[self.metaAgentID] = []
                episode_stop_ops[self.metaAgentID] = []


                # ===============================start training =======================================================================
                # RL
                if True:
                    # prepare to save GIF
                    saveGIF = False
                    global GIFS_FREQUENCY_RL
                    if OUTPUT_GIFS and self.agentID == 1 and ((not TRAINING) or (episode_count >= self.nextGIF)):
                        saveGIF = True
                        self.nextGIF = episode_count + GIFS_FREQUENCY_RL
                        GIF_episode = int(episode_count)
                        GIF_frames = [self.env._render()]

                    # start RL
                    self.env.finished = False
                    agent_done = False
                    while not self.env.finished:
                        if not agent_done:
                            # todo: add multi-output here
                            a_dist, v, rnn_state, \
                            blocking, meangoal, message = self.sess.run([self.local_AC.policy,
                                                                          self.local_AC.value,
                                                                          self.local_AC.state_out,
                                                                          self.local_AC.blocking,
                                                                          self.local_AC.mean_goal,
                                                                          self.local_AC.message,
                                                                                                ],
                                                                 # todo: feed the message(last time step) here
                                                        feed_dict={self.local_AC.inputs     : [s[0]],  # state
                                                                   self.local_AC.goal_pos   : [s[1]],  # goal vector
                                                                   self.local_AC.state_in[0]: rnn_state[0],
                                                                   self.local_AC.state_in[1]: rnn_state[1],
                                                                   })

                            skipping_state = False
                            train_policy = train_val = 1

                        if not skipping_state and not agent_done:
                            if not (np.argmax(a_dist.flatten()) in validActions):
                                episode_inv_count += 1
                                train_val = 0
                            train_valid = np.zeros(a_size)
                            train_valid[validActions] = 1

                            valid_dist = np.array([a_dist[0, validActions]])
                            valid_dist /= np.sum(valid_dist)

                            a = validActions[np.random.choice(range(valid_dist.shape[1]), p=valid_dist.ravel())]
                            joint_actions[self.metaAgentID][self.agentID] = a
                            if a == 0:
                                episode_stop_count += 1

                            # public the message here 'joint_comms'
                            joint_comms[self.metaAgentID][self.agentID] = message
                            joint_blocking[self.metaAgentID][self.agentID] = self.env.individual_blocking[self.agentID]

                        # Make A Single Agent Gather All Information

                        self.synchronize()

                        if self.agentID == 1:
                            # Add the message channel, set the communication channel while set the state.
                            all_obs, all_rewards = self.env.step_all(movement_dict=joint_actions[self.metaAgentID],
                                                                     msg_dict=joint_comms[self.metaAgentID])    # already contain the local comms_map

                            for i in range(1, self.num_workers+1):
                                joint_observations[self.metaAgentID][i] = all_obs[i]
                                joint_rewards[self.metaAgentID][i]      = all_rewards[i]
                                joint_done[self.metaAgentID][i]         = (self.env.world.agents[i].status >=1)
                            if saveGIF and self.agentID == 1:
                                GIF_frames.append(self.env._render())

                        self.synchronize()  # synchronize threads

                        # Get observation,reward, valid actions for each agent
                        s1           = joint_observations[self.metaAgentID][self.agentID]
                        r            = copy.deepcopy(joint_rewards[self.metaAgentID][self.agentID])

                        if not agent_done:
                            validActions = self.env.listValidActions(self.agentID, s1)


                        self.synchronize()
                        # Append to Appropriate buffers
                        if not skipping_state and not agent_done:
                            episode_buffer.append([s[0], a, joint_rewards[self.metaAgentID][self.agentID] , s1, v[0, 0], train_valid, s[1], s[2], joint_blocking[self.metaAgentID][self.agentID], meangoal ,blocking, message, train_val,train_policy])
                            episode_values.append(v[0, 0])
                        episode_reward += r
                        episode_step_count += 1

                        # Update State
                        s = s1

                        # If the episode hasn't ended, but the experience buffer is full, then we
                        # make an update step using that experience rollout.
                        if (not agent_done) and (len(episode_buffer)>1) and ((len(episode_buffer) % EXPERIENCE_BUFFER_SIZE == 0) or joint_done[self.metaAgentID][self.agentID] or episode_step_count==max_episode_length):
                            # Since we don't know what the true final return is,
                            # we "bootstrap" from our current value estimation.
                            if len(episode_buffer) >= EXPERIENCE_BUFFER_SIZE:
                                train_buffer = episode_buffer[-EXPERIENCE_BUFFER_SIZE:]
                            else:
                                train_buffer = episode_buffer[:]

                            # if joint_done[self.metaAgentID][self.agentID]:
                            #     s1Value        = 0       # Terminal state
                            #     episode_buffer = []
                            #     targets_done   += 1

                            # else:
                            s1Value = self.sess.run(self.local_AC.value,
                                                feed_dict={self.local_AC.inputs     : np.array([s[0]]),
                                                            self.local_AC.goal_pos   : [s[1]],
                                                            self.local_AC.state_in[0]: rnn_state[0],
                                                            self.local_AC.state_in[1]: rnn_state[1]})[0, 0]


                            self.loss_metrics, grads = self.calculateGradient(train_buffer, s1Value, episode_count, rnn_state0)

                            self.allGradients.append(grads)


                            rnn_state0 = rnn_state

                        self.synchronize()

                        # finish condition: reach max-len or all agents are done under one-shot mode
                        if episode_step_count >= max_episode_length:
                            break


                    episode_lengths[self.metaAgentID].append(episode_step_count)
                    episode_mean_values[self.metaAgentID].append(np.nanmean(episode_values))
                    episode_invalid_ops[self.metaAgentID].append(episode_inv_count)
                    episode_stop_ops[self.metaAgentID].append(episode_stop_count)
                    swarm_reward[self.metaAgentID] += episode_reward
                    swarm_targets[self.metaAgentID] += targets_done


                    self.synchronize()
                    if self.agentID == 1:
                        episode_rewards[self.metaAgentID].append(swarm_reward[self.metaAgentID])
                        episode_finishes[self.metaAgentID].append(swarm_targets[self.metaAgentID])

                        if saveGIF:
                            make_gif(np.array(GIF_frames),
                                     '{}/episode_{:d}_{:d}_{:.1f}.gif'.format(gifs_path,GIF_episode, episode_step_count,
                                                                           swarm_reward[self.metaAgentID]))

                    self.synchronize()


                    perf_metrics = np.array([
                        episode_step_count,
                        np.nanmean(episode_values),
                        episode_inv_count,
                        episode_stop_count,
                        episode_reward,
                        targets_done
                    ])

                    assert len(self.allGradients) > 0, 'Empty gradients at end of RL episode?!'
                    return perf_metrics



    def synchronize(self):
        # handy thing for keeping track of which to release and acquire
        if not hasattr(self, "lock_bool"):
            self.lock_bool = False
        self.groupLock.release(int(self.lock_bool), self.name)
        self.groupLock.acquire(int(not self.lock_bool), self.name)
        self.lock_bool = not self.lock_bool



    def work(self, currEpisode, coord, saver, allVariables):
        '''
        Interacts with the environment. The agent gets either gradients or experience buffer
        '''
        self.currEpisode = currEpisode


        if COMPUTE_TYPE == COMPUTE_OPTIONS.multiThreaded:
            self.perf_metrics = self.run_episode_multithreaded(currEpisode, coord)
        else:
            print("not implemented")
            assert(1==0)


        # gradients are accessed by the runner in self.allGradients
        return

    def parse_path(self, episode_count):
        """needed function to take the path generated from M* and create the
        observations and actions for the agent
        path: the exact path ouput by M*, assuming the correct number of agents
        returns: the list of rollouts for the "episode":
                list of length num_agents with each sublist a list of tuples
                (observation[0],observation[1],optimal_action,reward)"""

        global GIF_frames, SAVE_IL_GIF, IL_GIF_PROB
        saveGIF = False
        if np.random.rand() < IL_GIF_PROB:
            saveGIF = True
        if saveGIF and SAVE_IL_GIF:
            GIF_frames = [self.env._render()]
        result = [[] for i in range(self.num_workers)]
        msg = np.float32(0)
        blocking = np.float32(0)
        reward = np.float32(0)
        advantages = np.float32(0)
        meangoal = np.array([0., 0.], dtype='float32')

        actions = {}
        o = {}
        finished = {}
        train_imitation = {}
        count_finished = 0
        pos_buffer = []
        goal_buffer = []
        all_obs = self.env._observe()
        for agentID in range(1, self.num_workers + 1):
            o[agentID] = all_obs[agentID]
            train_imitation[agentID] = 1
            finished[agentID] = 0
        step_count = 0
        while step_count <= max_episode_length and count_finished < self.num_workers:
            path = self.env.expert_until_first_goal()
            if path is None:  # solution not exists
                if step_count != 0:
                    return result, 0
                print('Failed intially')
                return None, 0
            none_on_goal = True  # todo:
            path_step = 1
            while none_on_goal and step_count <= max_episode_length and count_finished < self.num_workers:
                positions = []
                goals = []
                for i in range(self.num_workers):
                    agent_id = i + 1
                    # if finished[agent_id]:  # todo:
                    #     actions[agent_id] = 0
                    # else:
                    #     next_pos = path[path_step][i]
                    #     diff = tuple_minus(next_pos, self.env.world.getPos(agent_id))
                    #     try:
                    #         actions[agent_id] = dir2action(diff)
                    #     except:
                    #         print('(parse_path)pos_buffer', pos_buffer)
                    #         print('(parse_path)goal_buffer', goal_buffer)
                    #         actions[agent_id] = dir2action(diff)
                    next_pos = path[path_step][i]           # todo
                    diff = tuple_minus(next_pos, self.env.world.getPos(agent_id))
                    try:
                        actions[agent_id] = dir2action(diff)
                    except:
                        print('(parse_path)pos_buffer', pos_buffer)
                        print('(parse_path)goal_buffer', goal_buffer)
                        actions[agent_id] = dir2action(diff)

                if ENV_DEBUG_MODE:
                    print('(parse_path)actions', actions)
                all_obs, _ = self.env.step_all(actions)

                for i in range(self.num_workers):
                    agent_id = i + 1
                    positions.append(self.env.world.getPos(agent_id))
                    goals.append(self.env.world.getGoal(agent_id))
                    result[i].append(
                        [o[agent_id][0], o[agent_id][1], o[agent_id][2], actions[agent_id], train_imitation[agent_id],
                         msg, blocking, reward, advantages])
                    if self.env.world.agents[agent_id].status >= 1 and finished[agent_id] != 1:
                        # none_on_goal = False # todo:
                        finished[agent_id] = 1  # todo:
                        count_finished += 1
                pos_buffer.append(positions)
                goal_buffer.append(goals)
                if saveGIF and SAVE_IL_GIF:
                    GIF_frames.append(self.env._render())
                o = all_obs
                step_count += 1
                path_step += 1
        if saveGIF and SAVE_IL_GIF:
            make_gif(np.array(GIF_frames),
                     '{}/episodeIL_{}.gif'.format(gifs_path, episode_count))
        return result, count_finished


    def shouldRun(self, coord, episode_count=None):
        if TRAINING:
            return not coord.should_stop()

