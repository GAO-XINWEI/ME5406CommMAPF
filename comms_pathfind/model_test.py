import numpy as np
import tensorflow as tf
import os
import ray
import sys

from Ray_ACNet import ACNet
from Runner import imitationRunner, RLRunner

from parameters import *
import random


load_model = True

ray.init(num_gpus=1)


tf.reset_default_graph()
print("Hello World")

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 1.0 / (NUM_META_AGENTS - NUM_IL_META_AGENTS + 1)
config.gpu_options.allow_growth = True


# Create directories
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)


global_step = tf.placeholder(tf.float32)

if ADAPT_LR:
    # computes LR_Q/sqrt(ADAPT_COEFF*steps+1)
    # we need the +1 so that lr at step 0 is defined
    lr = tf.divide(tf.constant(LR_Q), tf.sqrt(tf.add(1., tf.multiply(tf.constant(ADAPT_COEFF), global_step))))
else:
    lr = tf.constant(LR_Q)

def main():
    with tf.device("/gpu:0"):
        trainer = tf.contrib.opt.NadamOptimizer(learning_rate=lr, use_locking=True)
        global_network = ACNet(GLOBAL_NET_SCOPE,a_size,trainer,False,NUM_CHANNEL, OBS_SIZE,GLOBAL_NET_SCOPE, GLOBAL_NETWORK=True)
        saver = tf.train.Saver(max_to_keep=1)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        if load_model == True:
            print ('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(model_path)
            p=ckpt.model_checkpoint_path
            p=p[p.find('-')+1:]
            p=p[:p.find('.')]
            curr_episode=int(p)
            saver.restore(sess,ckpt.model_checkpoint_path)
            print("curr_episode set to ",curr_episode)


        # launch all of the threads:

        il_agents = [imitationRunner.remote(i) for i in range(NUM_IL_META_AGENTS)]
        rl_agents = [RLRunner.remote(i) for i in range(NUM_IL_META_AGENTS, NUM_META_AGENTS)]
        meta_agents = il_agents + rl_agents



        # get the initial weights from the global network
        weight_names = tf.trainable_variables()
        weights = sess.run(weight_names) # Gets weights in numpy arrays CHECK


        weightVars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)


        # launch the first job (e.g. getGradient) on each runner
        jobList = [] # Ray ObjectIDs
        for i, meta_agent in enumerate(meta_agents):
            jobList.append(meta_agent.job.remote(weights, curr_episode))
            curr_episode += 1

        IDs = [None] * NUM_META_AGENTS

        numImitationEpisodes = 0
        numRLEpisodes = 0
        try:
            while True:
                # wait for any job to be completed - unblock as soon as the earliest arrives
                done_id, jobList = ray.wait(jobList)

                # get the results of the task from the object store
                jobResults, metrics, info = ray.get(done_id)[0]

                if info['is_imitation']:
                    numImitationEpisodes += 1
                else:
                    numRLEpisodes += 1

                # get the updated weights from the global network
                weight_names = tf.trainable_variables()
                weights = sess.run(weight_names)
                curr_episode += 1

                # show 5 episodes for IL and RL, respectively
                print('numrl:', numRLEpisodes, 'NUMIL:', numImitationEpisodes)
                if numRLEpisodes >= 5 and numImitationEpisodes >= 5:
                    print('finish_test')
                    sys.exit()
                elif not info['is_imitation']:
                    meta_agents[info['id']].__del__.remote()
                    ray.kill(meta_agents[info['id']])
                    if numRLEpisodes < 5:
                        meta_agents[info['id']] = RLRunner.remote(info['id'])
                        jobList.extend([meta_agents[info['id']].job.remote(weights, curr_episode)])
                elif  info['is_imitation']:
                    if numImitationEpisodes < 5:
                        jobList.extend([meta_agents[info['id']].job.remote(weights, curr_episode)])
                    else:
                        meta_agents[info['id']].__del__.remote()
                        ray.kill(meta_agents[info['id']])


        except KeyboardInterrupt:
            print("CTRL-C pressed. killing remote workers")
            for a in meta_agents:
                ray.kill(a)


if __name__ == "__main__":
    main()
