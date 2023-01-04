'''
There are two ways to train after loading pre-trained LTL Module.
(1) One is to continue to update the weights of pre-trained LTL Module,
(2) and the other is to freeze the weights of pre-trained LTL Module.
It has been tested that freezing the weights of pre-trained LTL Module, the agent can still achieve good performance.
To further reduce the resources used, we use the pre-trained module to load the corresponding LTL formula,
and concat it with the states in the environment, which achieves the same effect as freeze the weights for direct training.

To generate load the vector of the corresponding LTL formula, the following code can be run:
>>> python trans_norm.py --num_encoder_layers 1 --d_out 16
'''

# Imports
import numpy as np

import torch
from tensorboardX import SummaryWriter

from schedules import LinearSchedule
from dfa import *
from envs.game import *
import random, time, os.path, shutil, copy, glob

import utils
import ltl_progression
from ReplayBuffer import ContextBuffer

from algos.TL2Trans import BaselineDTQN

from collections import deque

"""
This baseline solves the problem using standard q-learning over the cross product 
between the LTL instruction and the MDP
"""

def prYellow(prt): print("\033[93m {}\033[00m".format(prt))
def prGreen(prt): print("\033[92m {}\033[00m".format(prt))

def progression(ltl_formula, truth_assignment):
    result_ltl = ltl_progression.progress_and_clean(ltl_formula, truth_assignment)
    return result_ltl

def _run_DQN(algorithm, task_params, progressed_tasks, task_results, buffer4tasks,
             tester,  curriculum, args, show_print):
    # state_txt = "current_state.txt"
    writer = SummaryWriter(f'./logs/craft/TL2/cont:{args.history_length}_dout:{args.d_out}/')

    learning_params = tester.learning_params
    testing_params = tester.testing_params
    total_steps = 0
    default_total_step = args.warm_up
    exploration = LinearSchedule(schedule_timesteps=int(learning_params.exploration_fraction * default_total_step),
                                 initial_p=1.0, final_p=learning_params.exploration_final_eps)
    for i_episode in range(learning_params.max_episodes):

        # Initializing parameters
        init_LTL = task_params.ltl_task
        episode_steps = 0
        training_reward = 0
        reward = 0
        done = False

        # Initializing parameters
        task = Game(task_params)
        actions = task.get_actions()

        # env reset
        num_states = len(task.get_features())
        s1 = np.concatenate([task.get_features(), np.array(task_results[init_LTL])])

        ### history ####
        rewards_hist = deque(maxlen=args.history_length)
        actions_hist = deque(maxlen=args.history_length)
        obsvs_hist = deque(maxlen=args.history_length)

        next_hrews = deque(maxlen=args.history_length)
        next_hacts = deque(maxlen=args.history_length)
        next_hobvs = deque(maxlen=args.history_length)

        # zero_action = np.zeros(num_actions)
        zero_obs = np.zeros(len(task.get_features())+args.d_out)
        for _ in range(args.history_length):
            rewards_hist.append(0)
            actions_hist.append(0)
            obsvs_hist.append(zero_obs)
            next_hrews.append(0)
            next_hacts.append(0)
            next_hobvs.append(zero_obs)

        rewards_hist.append(0)
        random_action = np.random.randint(0, 4, 1)
        actions_hist.append(random_action[0])
        obsvs_hist.append(s1)

        # f = open(state_txt, 'a')

        # Starting interaction with the environment
        while not done:

            np_pre_actions = np.asarray(actions_hist, dtype=np.float32).flatten()  # (hist, action_dim) => (hist *action_dim,)
            np_pre_rewards = np.asarray(rewards_hist, dtype=np.float32)  # (hist, )
            np_pre_obsers = np.asarray(obsvs_hist, dtype=np.float32).flatten()  # (hist, action_dim) => (hist *action_dim,)

            # Choosing an action to perform
            if random.random() < exploration.value(total_steps):
                a = random.choice(actions)
            else:
                a = Actions(algorithm.get_best_action(s1, torch.FloatTensor(np_pre_actions).view(1, len(np_pre_actions)),
                                                    torch.FloatTensor(np_pre_rewards).view(1, len(np_pre_rewards)),
                                                    torch.FloatTensor(np_pre_obsers).view(1, len(np_pre_obsers))))

            # f.write(f"{episode_steps} step is {s1}\n")

            # Executing the action
            task.execute_action(a)
            true_props = task.get_true_propositions()
            progressed_ltl = progression(init_LTL, true_props)
            if progressed_ltl == 'True':
                reward = 1
            elif progressed_ltl == 'False':
                reward = -1
            else:
                reward = 0
            training_reward += reward
            init_LTL = progressed_ltl

            ###############
            next_hrews.append(reward)
            next_hacts.append(a.value)
            next_hobvs.append(s1)

            # np_next_hacts and np_next_hrews are required for TD3 alg
            np_next_hacts = np.asarray(next_hacts, dtype=np.float32).flatten()  # (hist, action_dim) => (hist *action_dim,)
            np_next_hrews = np.asarray(next_hrews, dtype=np.float32)  # (hist, )
            np_next_hobvs = np.asarray(next_hobvs, dtype=np.float32).flatten()  # (hist, )

            # Saving this transition
            s2 = np.concatenate([task.get_features(), np.array(task_results[progressed_ltl])])

            done = True if reward == 1 or reward == -1 or task.check_env_done() else False

            episode_steps += 1
            total_steps += 1

            # counter example experience
            for each_task in progressed_tasks:
                t_s1 = np.concatenate([s1[:num_states], np.array(task_results[each_task])])
                t_progressed_ltl = progression(each_task, true_props)
                t_s2 = np.concatenate([s2[:num_states], np.array(task_results[t_progressed_ltl])])
                if t_progressed_ltl == 'True':
                    t_reward = 1.0
                elif t_progressed_ltl == 'False':
                    t_reward = -1.0
                else:
                    t_reward = 0.
                t_done = True if t_reward == 1.0 or t_reward == -1.0 or task.check_env_done() else False
                buffer4tasks[each_task].add(t_s1, a.value, t_reward, t_s2, t_done,
                                            np_pre_actions, np_pre_rewards, np_pre_obsers,
                                            np_next_hacts, np_next_hrews, np_next_hobvs)

            # new becomes old
            rewards_hist.append(reward)
            actions_hist.append(a.value)
            obsvs_hist.append(s1)

            s1 = s2

            # Learning
            if total_steps > learning_params.learning_starts and total_steps % learning_params.train_freq == 0:
                S1, A, R, S2, DONE, PA, PR, PX, NA, NR, NX = [], [], [], [], [], [], [], [], [], [], []
                for i in buffer4tasks:
                    p_S1, p_A, p_R, p_S2, p_DONE, p_pa, p_pr, p_px, p_na, p_nr, p_nx = buffer4tasks[i].sample(learning_params.batch_size)
                    S1.extend(p_S1)
                    A.extend(p_A)
                    R.extend(p_R)
                    S2.extend(p_S2)
                    DONE.extend(p_DONE)
                    PA.extend(p_pa)
                    PR.extend(p_pr)
                    PX.extend(p_px)
                    NA.extend(p_na)
                    NR.extend(p_nr)
                    NX.extend(p_nx)
                algorithm.learn(np.array(S1), np.array(A), np.array(R), np.array(S2), np.array(DONE),
                                np.array(PA), np.array(PR), np.array(PX),
                                np.array(NA), np.array(NR), np.array(NX),
                                )

            # Updating the target network
            if total_steps > learning_params.learning_starts and total_steps % learning_params.target_network_update_freq == 0:
                # Update target network periodically.
                algorithm.update_target_network()

        normalized_reward = (learning_params.gamma**(episode_steps-1))/tester.optimal[task_params.ltl_task] + training_reward
        # normalized_reward = (0.9 ** (episode_steps - 1)) / tester.optimal[init_LTL] + training_reward
        if reward == -1 and normalized_reward > 1.0:
            normalized_reward = -1.
        writer.add_scalar('Hidden:{}_Seed:{}/each_epi'.format(args.dqn_hidden, args.seed), normalized_reward, i_episode)
        prYellow("Episode: {}| episodes_steps: {}| total_steps: {}| rewards: {}| performace: {}".format(i_episode,
                                                                                                        episode_steps,
                                                                                                        total_steps,
                                                                                                        training_reward,
                                                                                                  round(normalized_reward, 4)))
        # f.close()


def get_progressed_tasks(tasks, d_out):
    progressed_task_dict = {}

    tempo_task_0 = ('until', ('not', 'n'), ('and', 'a', (
        'until', ('not', 'n'), ('and', 'b', ('until', ('not', 'n'), ('and', 'c', ('until', ('not', 'n'), 'd')))))))
    tempo_task_1 = ('until', ('not', 'n'), (
        'or', ('and', 'b', ('until', ('not', 'n'), ('and', 'c', ('until', ('not', 'n'), 'd')))), ('and', 'a', (
            'until', ('not', 'n'), ('and', 'b', ('until', ('not', 'n'), ('and', 'c', ('until', ('not', 'n'), 'd'))))))))
    tempo_task_2 = ('until', ('not', 'n'), ('or', ('or', ('and', 'c', ('until', ('not', 'n'), 'd')), (
        'and', 'b', ('until', ('not', 'n'), ('and', 'c', ('until', ('not', 'n'), 'd'))))), ('and', 'a', (
        'until', ('not', 'n'), ('and', 'b', ('until', ('not', 'n'), ('and', 'c', ('until', ('not', 'n'), 'd'))))))))
    tempo_task_3 = ('until', ('not', 'n'), ('or', ('or', ('or', 'd', ('and', 'c', ('until', ('not', 'n'), 'd'))), (
        'and', 'b', ('until', ('not', 'n'), ('and', 'c', ('until', ('not', 'n'), 'd'))))), ('and', 'a', (
        'until', ('not', 'n'), ('and', 'b', ('until', ('not', 'n'), ('and', 'c', ('until', ('not', 'n'), 'd'))))))))

    if d_out == 8:
        progressed_task_dict[tempo_task_0] = [-0.9980, 0.9980, -0.9993, 0.9774, 0.1786, -0.9901, 0.9974, 0.9985]
        progressed_task_dict[tempo_task_1] = [-0.9988, -0.9990, -0.9987, 0.9589, 0.9719, 0.9800, 0.9990, 0.2763]
        progressed_task_dict[tempo_task_2] = [0.9977, -0.9957, -0.9997, 0.2185, -0.2558, -0.9858, -0.9794, 0.9833]
        progressed_task_dict[tempo_task_3] = [0.9986, -0.9733, -0.9809, -0.9869, -0.9911, -0.8958, 0.9929, 0.9973]
        progressed_task_dict['True'] = [1., 1., 1., 1., 1., 1., 1., 1.]
        progressed_task_dict['False'] = [-1., -1., -1., -1., -1., -1., -1., -1.]
    elif d_out == 12:
        progressed_task_dict[tempo_task_0] = [-0.9907, 0.9931, -0.9965, -0.9997, 0.9940, -0.9985, -0.9959, 0.9975,
                                              0.9950, -0.9966, 0.9974, 0.9955]
        progressed_task_dict[tempo_task_1] = [0.4605, -0.9838, -0.9998, -0.9993, -0.9898, -0.9976, 0.9951, 0.9968,
                                              0.9915, 0.9966, 0.9998, 0.9999]
        progressed_task_dict[tempo_task_2] = [0.3560, -0.9995, -0.9999, 0.9962, -0.9973, 0.9889, 0.9979, 0.9987,
                                              -0.9787, 0.9557, 0.9982, 0.9998]
        progressed_task_dict[tempo_task_3] = [0.9864, 0.9981, -0.9994, 0.9954, 0.4421, -0.6423, 0.9971, 0.9994,
                                              -0.8758, 0.9998, 0.9988, 0.9988]
        progressed_task_dict['True'] = [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]
        progressed_task_dict['False'] = [-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.]
    elif d_out == 16:
        progressed_task_dict[tempo_task_0] = [-0.9870, -0.9965, 0.9895, 0.9826, 0.9983, -0.9776, -0.9702, -0.5873,
                                              -0.9961, 0.9996, 0.9991, -0.9950, 0.9899, -0.9965, -0.9948, -0.9958]
        progressed_task_dict[tempo_task_1] = [0.9810, -0.9982, -0.9582, 0.9958, -0.9994, 0.9686, 0.9953, -0.9947,
                                              0.9953, 0.9981, 0.9976, -0.9807, 0.9813, 0.9996, 0.9997, -0.9868]
        progressed_task_dict[tempo_task_2] = [0.9969, -0.9983, 0.9879, 0.9996, -1.0000, 0.9937, -0.9990, -0.9998,
                                              0.9878, -0.9924, -0.9993, -0.9922, 0.9924, 0.9994, -0.9922, -0.9568]
        progressed_task_dict[tempo_task_3] = [0.9914, -0.9966, 0.9997, 0.9969, -0.9973, -0.8476, -0.9997, -0.9942,
                                              0.9958, 0.9955, -0.9946, 0.1987, 0.9657, -0.9989, -0.9939, 0.9620]
        progressed_task_dict['True'] = [1., 1., 1., 1., 1., 1., 1., 1.,
                                        1., 1., 1., 1., 1., 1., 1., 1.]
        progressed_task_dict['False'] = [-1., -1., -1., -1., -1., -1., -1., -1.,
                                         -1., -1., -1., -1., -1., -1., -1., -1.]
    else:
        progressed_task_dict[tempo_task_0] = [-1.0000, -0.9995, -0.9321, -0.7166]
        progressed_task_dict[tempo_task_1] = [-0.9986, -0.7673, 0.9795, -0.9976]
        progressed_task_dict[tempo_task_2] = [-0.1383, 0.9818, 0.9996, -0.9996]
        progressed_task_dict[tempo_task_3] = [-0.9999, 0.9990, -0.7781, -0.8322]
        progressed_task_dict['True'] = [1., 1., 1., 1.]
        progressed_task_dict['False'] = [-1., -1., -1., -1.]

    subtask_list = []
    subtask_list.append(tempo_task_0)
    subtask_list.append(tempo_task_1)
    subtask_list.append(tempo_task_2)
    subtask_list.append(tempo_task_3)

    return subtask_list, progressed_task_dict

def run_experiments(alg_name, tester, curriculum, saver, num_times, args, show_print):
    # Running the tasks 'num_times'
    time_init = time.time()
    learning_params = tester.learning_params
    for t in range(num_times):
        # Setting the random seed to 't'
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)

        progressed_tasks, task_results = get_progressed_tasks(tester.tasks, args.d_out)
        buffer4tasks = {}
        for i in progressed_tasks:
            buffer4tasks[i] = ContextBuffer(learning_params.buffer_size)

        curriculum.restart()
        task_aux = Game(tester.get_task_params(curriculum.get_current_task()))
        num_features = len(task_aux.get_features())
        num_actions = len(task_aux.get_actions())
        num_ltl_state = args.d_out

        # Initializing algorithm
        # todo: check
        algorithm = BaselineDTQN(num_actions, num_features, num_ltl_state, learning_params, args)

        if args.pretrain:
            pre_default_dir = f"./pretrain/Transformer-dumb_ac_Eventually_1_5_1_4_Simple-LTL-Env-v0_seed:1_n_encoder:{args.num_encoder_layers}_nhead:8_d_model:64_dim_feedforward:256_d_out:{args.d_out}_n{args.num_encoder_layers}_dout_{args.d_out}_ch/train"
            model_dirs = glob.glob(pre_default_dir)
            pretrained_model_dir = model_dirs[0]
            pretrained_status = utils.get_status(pretrained_model_dir, args)
            algorithm.to_load_pretrain_LTL(pretrained_status["model_state"])
            prYellow("Pretrained model loaded.\n")
            algorithm.update_target_network()

        # Reseting default values
        curriculum.restart()

        # Running the tasks

        task = curriculum.get_next_task()
        print('LTL task is {}'.format(task))
        task_params = tester.get_task_params(task)
        _run_DQN(algorithm, task_params, progressed_tasks, task_results,
                 buffer4tasks, tester, curriculum, args, show_print)

    print("Time:", "%0.2f" % ((time.time() - time_init) / 60), "mins")
