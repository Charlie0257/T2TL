# Imports
import argparse, LTL2Trans, TL1, TL2
from test_utils import Tester, Saver, TestingParameters
from curriculum import CurriculumLearner


class LearningParameters:
    def __init__(self, args):

        self.lr = args.lr
        self.max_timesteps_per_task = args.max_timesteps_per_task
        self.buffer_size = args.buffer_size
        self.exploration_fraction = args.exploration_fraction
        self.exploration_final_eps = args.exploration_final_eps
        self.train_freq = args.train_freq
        self.batch_size = args.batch_size
        self.print_freq = 1000
        self.learning_starts = args.learning_starts
        self.gamma = args.gamma
        # self.rs_gamma = args.rs_gamma
        # self.gamma_B = args.gamma_B
        self.target_network_update_freq = args.target_network_update_freq
        self.max_episodes = args.max_episodes
        self.steps_per_epi = args.steps_per_epi


def run_experiment(alg_name, map_id, tasks_id, num_times, r_good, args, show_print):
    # configuration of testing params
    testing_params = TestingParameters()

    # configuration of learning params
    learning_params = LearningParameters(args)

    # Setting the experiment
    tester = Tester(learning_params, testing_params, map_id, tasks_id)

    # Setting the curriculum learner
    curriculum = CurriculumLearner(tester.tasks, r_good=r_good)

    # Setting up the saver
    saver = Saver(alg_name, tester, curriculum)

    # Transformer
    if alg_name == "Transformer":
        LTL2Trans.run_experiments(alg_name, tester, curriculum, saver, num_times, args, show_print)

    # TL1
    if alg_name == "T1TL":
        TL1.run_experiments(alg_name, tester, curriculum, saver, num_times, args, show_print)

    # TL2
    if alg_name == "T2TL":
        TL2.run_experiments(alg_name, tester, curriculum, saver, num_times, args, show_print)



def run_single_experiment(alg, tasks_id, map_id, args):
    num_times = 1
    show_print = True
    r_good = 0.8 if tasks_id == 2 else 0.9

    print("Running", "r_good:", r_good, "alg:", alg, "map_id:", map_id, "tasks_id:", tasks_id)
    run_experiment(alg, map_id, tasks_id, num_times, r_good, args, show_print)


if __name__ == "__main__":

    # EXAMPLE: python3 run_experiments.py --algorithm="lpopl" --tasks="sequence" --map=0
    # Getting params
    algorithms = ["Transformer", "T1TL", "T2TL"]
    tasks = ["Sequential4Minicraft"]

    parser = argparse.ArgumentParser(prog="run_experiments",
                                     description='Runs a multi-task RL experiment over a gridworld domain that is inspired by Minecraft.')
    # Comet ML
    parser.add_argument('--log_comet', action='store_true', dest='log_comet',
                        default=False, help="Whether to log data")
    parser.add_argument('--comet_key', default='', help='Comet API key')
    parser.add_argument('--comet_workspace', default='', help='Comet workspace')

    parser.add_argument('--algorithm', default='T1TL', type=str,
                        help='This parameter indicated which RL algorithm to use. The options are: ' + str(algorithms))
    parser.add_argument('--tasks', default='Sequential4Minicraft', type=str,
                        help='This parameter indicated which tasks to solve. The options are: ' + str(tasks))
    parser.add_argument('--map', default=2, type=int,
                        help='This parameter indicated which map to use. It must be a number between -1 and 9. Use "-1" to run experiments over the 10 maps, 3 times per map')
    parser.add_argument('--seed', default=0, type=int, help='')


    parser.add_argument("--pretrain", type=bool, default=True,
                        help="whether to use the pretrained LTL model or not")
    parser.add_argument("--freeze_pretrained_params", type=bool, default=True,
                        help="whether to freeze pretrained LTL params or not")


    # DQN special parameters
    parser.add_argument("--lr", type=float, default=0.0001, help="")
    parser.add_argument("--max_timesteps_per_task", type=int, default=50000, help="used in multi-task setting")
    parser.add_argument("--buffer_size", type=int, default=50000, help="")
    parser.add_argument("--exploration_fraction", type=float, default=0.1, help="")
    parser.add_argument("--exploration_final_eps", type=float, default=0.02, help="")
    parser.add_argument("--train_freq", type=int, default=1, help="")
    parser.add_argument("--batch_size", type=int, default=64, help="")
    parser.add_argument("--learning_starts", type=int, default=1000, help="")
    parser.add_argument("--gamma", type=float, default=0.9, help="")
    parser.add_argument("--rs_gamma", type=float, default=0.90, help="")
    parser.add_argument("--target_network_update_freq", type=int, default=100, help="")
    parser.add_argument("--dqn_hidden", type=int, default=256, help="")
    parser.add_argument('--buffer', default='mean', type=str, help='whether to use mean or together')
    parser.add_argument('--max_episodes', default=2000, type=int, help='')
    parser.add_argument('--steps_per_epi', default=250, type=int, help='')
    parser.add_argument('--warm_up', default=75000, type=int, help='')

    # Transformer special parameters
    parser.add_argument("--d_model", type=int, default=64, help="")
    parser.add_argument("--nhead", type=int, default=8, help="")
    parser.add_argument("--num_encoder_layers", type=int, default=1, help="")
    parser.add_argument("--pool", type=str, default='mean', help="")
    parser.add_argument("--dim_feedforward", type=int, default=256, help="")
    parser.add_argument("--dropout", type=float, default=0.0, help="")
    parser.add_argument("--d_out", type=int, default=16, help="")
    parser.add_argument("--layer_norm_eps", type=float, default=1e-5, help="")

    # The Context variable
    parser.add_argument("--use_cont", type=bool, default=True, help="")
    parser.add_argument("--history_length", type=int, default=12, help="")

    # Special4RM
    parser.add_argument("--RM_RS", type=bool, default=False, help="")
    parser.add_argument("--TL3_RS", type=bool, default=False, help="")

    parser.add_argument('--device', default="cuda:0", type=str, help="choose to use cuda:X or cpu")

    args = parser.parse_args()
    if args.algorithm not in algorithms: raise NotImplementedError(
        "Algorithm " + str(args.algorithm) + " hasn't been implemented yet")
    if args.tasks not in tasks: raise NotImplementedError("Tasks " + str(args.tasks) + " hasn't been defined yet")
    if not (-1 <= args.map < 10): raise NotImplementedError("The map must be a number between -1 and 9")

    # Running the experiment
    alg = args.algorithm
    tasks_id = tasks.index(args.tasks)
    map_id = args.map

    run_single_experiment(alg, tasks_id, map_id, args)
