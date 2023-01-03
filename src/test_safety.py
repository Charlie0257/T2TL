import argparse
import time
import sys
import numpy as np
import glfw
import utils
import torch

import gym
import safety_gym
import ltl_wrappers
import ltl_progression
from gym import wrappers, logger
from envs.safety import safety_wrappers

import csv

class RandomAgent(object):
    """This agent picks actions randomly"""
    def __init__(self, action_space):
        self.action_space = action_space

    def get_action(self, obs):
        return self.action_space.sample()

class PlayAgent(object):
    """
    This agent allows user to play with Safety's Point agent.

    Use the UP and DOWN arrows to move forward and back and
    use '<' and '>' to rotate the agent.
    """
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.prev_act = np.array([0, 0])
        self.last_obs = None

    def get_action(self, obs):
        # obs = obs["features"]

        key = self.env.key_pressed

        if(key == glfw.KEY_COMMA):
            current = np.array([0, 0.4])
        elif(key == glfw.KEY_PERIOD):
            current = np.array([0, -0.4])
        elif(key == glfw.KEY_UP):
            current = np.array([0.1, 0])
        elif(key == glfw.KEY_DOWN):
            current = np.array([-0.1, 0])
        elif(key == -1): # This is glfw.RELEASE
            current = np.array([0, 0])
            self.prev_act = np.array([0, 0])
        else:
            current = np.array([0, 0])

        self.prev_act = np.clip(self.prev_act + current, -1, 1)

        return self.prev_act

def run_policy(agent, env, max_ep_len=None, num_episodes=100, render=True):
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(1) #########

    # traj recorder
    position_track = 'track4Zone.csv'
    header_list = ['Steps', 'X_pos', 'Y_pos', 'Z_pos', 'End']
    csv_pos = open(position_track, mode="w", encoding="utf-8-sig", newline="")
    csv_pos = csv.writer(csv_pos)
    csv_pos.writerow(header_list)

    o, r, d, ep_ret, ep_cost, ep_len, n, info = env.reset(), 0, False, 0, 0, 0, 0, {'proposition':''}
    while n < num_episodes:
        if render:
            env.render()
            time.sleep(1e-3)

        # traj recorder
        track_list = [ep_len,
                      env.world.robot_pos()[0].item(),
                      env.world.robot_pos()[1].item(),
                      0.3,
                      info['proposition']]
        csv_pos.writerow(track_list)

        ltl_goal = ltl_progression.spotify(env.ltl_goal)
        env.show_text(ltl_goal.to_str())
        if("progress_info" in o.keys()):
            env.show_prog_info(o["progress_info"])

        a = agent.get_action(o)
        a = np.clip(a, env.action_space.low, env.action_space.high)
        o, r, d, info = env.step(a)
        ep_ret += r
        ep_cost += info.get('cost', 0)
        ep_len += 1

        if d or (ep_len == max_ep_len):
            track_list = [ep_len,
                          env.world.robot_pos()[0].item(),
                          env.world.robot_pos()[1].item(),
                          0.3,
                          info['proposition']]
            csv_pos.writerow(track_list)
            o, r, d, ep_ret, ep_cost, ep_len = env.reset(), 0, False, 0, 0, 0
            n += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    subparsers = parser.add_subparsers(dest='command')

    parser.add_argument('env_id', default='SafexpTest-v0', help='Select the environment to run')

    ## General parameters
    parser.add_argument("--algo", default='ppo',
                        help="algorithm to use: a2c | ppo (REQUIRED)")
    parser.add_argument("--env", default='Zones-5-v0',
                        help="name of the environment to train on (REQUIRED)")
    parser.add_argument("--ltl-sampler", default="Until_1_2_1_1",
                        help="the ltl formula template to sample from (default: DefaultSampler)")
    parser.add_argument("--model", default=None,
                        help="name of the model (default: {ENV}_{SAMPLER}_{ALGO}_{TIME})")
    parser.add_argument("--seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--log-interval", type=int, default=1,
                        help="number of updates between two logs (default: 10)")
    parser.add_argument("--save-interval", type=int, default=2,
                        help="number of updates between two saves (default: 10, 0 means no saving)")
    parser.add_argument("--procs", type=int, default=16,
                        help="number of processes (default: 16)")
    parser.add_argument("--frames", type=int, default=2 * 10 ** 7,
                        help="number of frames of training (default: 2*10e8)")
    parser.add_argument("--checkpoint-dir", default=None)

    ## Evaluation parameters
    parser.add_argument("--eval", action="store_true", default=False,
                        help="evaluate the saved model (default: False)")
    parser.add_argument("--eval-episodes", type=int, default=5,
                        help="number of episodes to evaluate on (default: 5)")
    parser.add_argument("--eval-env", default=None,
                        help="name of the environment to train on (default: use the same \"env\" as training)")
    parser.add_argument("--ltl-samplers-eval", default=None, nargs='+',
                        help="the ltl formula templates to sample from for evaluation (default: use the same \"ltl-sampler\" as training)")
    parser.add_argument("--eval-procs", type=int, default=1,
                        help="number of processes (default: use the same \"procs\" as training)")

    ## Parameters for main algorithm

    parser.add_argument("--epochs", type=int, default=10,
                        help="number of epochs for PPO (default: 4)")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="batch size for PPO (default: 256)")
    parser.add_argument("--frames-per-proc", type=int, default=4096,
                        help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
    parser.add_argument("--discount", type=float, default=0.998,
                        help="discount factor (default: 0.99)")
    parser.add_argument("--lr", type=float, default=0.0003,
                        help="learning rate (default: 0.0003)")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
    parser.add_argument("--entropy-coef", type=float, default=0.003,
                        help="entropy term coefficient (default: 0.01)")
    parser.add_argument("--value-loss-coef", type=float, default=0.5,
                        help="value loss term coefficient (default: 0.5)")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="maximum norm of gradient (default: 0.5)")
    parser.add_argument("--optim-eps", type=float, default=1e-8,
                        help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
    parser.add_argument("--optim-alpha", type=float, default=0.99,
                        help="RMSprop optimizer alpha (default: 0.99)")
    parser.add_argument("--clip-eps", type=float, default=0.2,
                        help="clipping epsilon for PPO (default: 0.2)")
    parser.add_argument("--ignoreLTL", action="store_true", default=False,
                        help="the network ignores the LTL input")
    parser.add_argument("--noLTL", action="store_true", default=False,
                        help="the environment no longer has an LTL goal. --ignoreLTL must be specified concurrently.")
    parser.add_argument("--progression-mode", default="full",
                        help="Full: uses LTL progression; partial: shows the propositions which progress or falsify the formula; none: only original formula is seen. ")
    parser.add_argument("--recurrence", type=int, default=1,
                        help="number of time-steps gradient is backpropagated (default: 1). If > 1, a LSTM is added to the model to have memory.")
    parser.add_argument("--gnn", default="Transformer", help="use gnn to model the LTL (only if ignoreLTL==True)")
    parser.add_argument("--trans_layer", type=int, default=1, help="the number of Transformer layers need to use")
    parser.add_argument("--int-reward", type=float, default=0.0,
                        help="the intrinsic reward for LTL progression (default: 0.0)")
    parser.add_argument("--pretrained-gnn", action="store_true", default=False, help="load a pre-trained LTL module.")
    parser.add_argument("--dumb-ac", action="store_true", default=False, help="Use a single-layer actor-critic")
    parser.add_argument("--freeze-ltl", action="store_true", default=False,
                        help="Freeze the gradient updates of the LTL module")

    # Transformer special parameters
    parser.add_argument("--d_model", type=int, default=64, help="")
    parser.add_argument("--nhead", type=int, default=8, help="")
    parser.add_argument("--num_encoder_layers", type=int, default=2, help="")
    parser.add_argument("--pool", type=str, default='mean', help="")
    parser.add_argument("--dim_feedforward", type=int, default=256, help="")
    parser.add_argument("--dropout", type=float, default=0.0, help="")
    parser.add_argument("--d_out", type=int, default=16, help="")
    parser.add_argument("--layer_norm_eps", type=float, default=1e-5, help="")

    parser.add_argument("--cuda", type=str, default='cuda:0', help="")

    parser_play = subparsers.add_parser('play',   help='A playable agent that can be controlled.')
    parser_random = subparsers.add_parser('random', help='An agent that picks actions at random (for testing).')
    parser_viz = subparsers.add_parser('viz', help='Load the agent model from a file and visualize its action on the env.')

    parser_viz.add_argument('model_path', type=str, help='The path to the model to load.')

    parser_viz.add_argument("--ltl-sampler", default="Default",
                    help="the ltl formula template to sample from (default: DefaultSampler)")


    args = vars(parser.parse_args()) # make it a dictionary
    args_2 = parser.parse_args()

    outdir = './storage/random-agent-results'

    if (args["command"] == "play"):
        env = gym.make(args["env_id"])
        env.num_steps = 10000000
        env = safety_wrappers.Play(env)
        env = ltl_wrappers.LTLEnv(env, ltl_sampler="Default")

        agent = PlayAgent(env)

    elif (args["command"] == "random"):
        env = gym.make(args["env_id"])
        env.num_steps = 10000
        env = safety_wrappers.Play(env)
        env = ltl_wrappers.LTLEnv(env, ltl_sampler="Default")

        agent = RandomAgent(env.action_space)

    elif (args["command"] == "viz"):
        # If the config is available (from trainig) then just load it here instead of asking the user of this script to provide all training time configs
        config = vars(utils.load_config(args["model_path"]))
        args.update(config)

        env = gym.make(args["env_id"])
        env = safety_wrappers.Play(env)
        env = ltl_wrappers.LTLEnv(env, ltl_sampler=args["ltl_sampler"], progression_mode=args["progression_mode"])

        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        agent = utils.Agent(env, env.observation_space, env.action_space, args["model_path"],
                args["ignoreLTL"], args["progression_mode"], args["gnn"], device=device, dumb_ac=args["dumb_ac"], args=args_2)
    else:
        print("Incorrect command: ", args["command"])
        exit(1)

    run_policy(agent, env, max_ep_len=30000, num_episodes=1000)

