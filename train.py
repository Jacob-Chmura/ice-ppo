from common.env.atari_wrappers import *
from common.env.procgen_wrappers import *
from common.logger import Logger
from common.storage import Storage
from common.model import NatureModel, ImpalaModel, ActorCritic
from common.policy import CategoricalPolicy
from common import set_global_seeds, set_global_log_levels

import os, time, yaml, argparse
import gym
#from procgen import ProcgenEnv
import random
import torch


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name',         type=str, default = 'bass_ice', help='experiment name')
    parser.add_argument('--env_name',         type=str, default = 'MontezumaRevenge-v0', help='environment ID')
    parser.add_argument('--start_level',      type=int, default = int(0), help='start-level for environment')
    parser.add_argument('--num_levels',       type=int, default = int(500), help='number of training levels for environment')
    parser.add_argument('--distribution_mode',type=str, default = 'hard', help='distribution mode for environment')
    parser.add_argument('--param_name',       type=str, default = 'hard', help='hyper-parameter ID')
    parser.add_argument('--device',           type=str, default = 'gpu', required = False, help='whether to use gpu')
    parser.add_argument('--gpu_device',       type=int, default = int(0), required = False, help = 'visible device in CUDA')
    parser.add_argument('--num_timesteps',    type=int, default = int(1000000000), help = 'number of training timesteps')
    parser.add_argument('--seed',             type=int, default = random.randint(0,9999), help='Random generator seed')
    parser.add_argument('--log_level',        type=int, default = int(40), help='[10,20,30,40]')
    parser.add_argument('--num_checkpoints',  type=int, default = int(1000), help='number of checkpoints to store')
    parser.add_argument('--loss',             type=str, default = 'entropy')
    parser.add_argument('--beta',             type=float, default = 0.5)
    parser.add_argument('--reset_on_extrinsic', type=bool, default=True)
    parser.add_argument('--exploration_param_name', type=str, default = 'bass_ice', help='exploration hyper-parameter ID')

    args = parser.parse_args()
    exp_name = args.exp_name
    env_name = args.env_name
    start_level = args.start_level
    num_levels = args.num_levels
    distribution_mode = args.distribution_mode
    param_name = args.param_name
    device = args.device
    gpu_device = args.gpu_device
    num_timesteps = args.num_timesteps
    seed = args.seed
    log_level = args.log_level
    num_checkpoints = args.num_checkpoints
    loss = args.loss
    beta = args.beta

    set_global_seeds(seed)
    set_global_log_levels(log_level)

    ####################
    ## HYPERPARAMETERS #
    ####################
    print('[LOADING HYPERPARAMETERS...]')
    with open('hyperparams/procgen/config.yml', 'r') as f:
        hyperparameters = yaml.safe_load(f)[param_name]
    for key, value in hyperparameters.items():
        print(key, ':', value)

    print('[LOADING EXPLORATION HYPERPARAMETERS...]')
    with open('hyperparams/exploration/exploration_config.yml', 'r') as f:
        exploration_config = yaml.safe_load(f)[args.exploration_param_name]
    for k, v in exploration_config.items():
        print(f"{k}: {v}")

    ############
    ## DEVICE ##
    ############
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
    device = torch.device('cuda')

    #################
    ## ENVIRONMENT ##
    #################
    print('INITIALIZAING ENVIRONMENTS...')
    n_steps = hyperparameters.get('n_steps', 256)
    n_envs = hyperparameters.get('n_envs', 64)
    # By default, pytorch utilizes multi-threaded cpu
    # Procgen is able to handle thousand of steps on a single core
    torch.set_num_threads(1)
    if env_name == "starpilot":
        env = ProcgenEnv(num_envs=n_envs,
                         env_name=env_name,
                         start_level=start_level,
                         num_levels=num_levels,
                         distribution_mode=distribution_mode)
        normalize_rew = hyperparameters.get('normalize_rew', True)
        env = VecExtractDictObs(env, "rgb")
        if normalize_rew:
            env = VecNormalize(env, ob=False) # normalizing returns, but not the img frames.
        env = TransposeFrame(env)
        env = ScaledFloatFrame(env)
    else:

        def _create_env(args_):
            env_ = gym.make(args_.env_name)
            if args_.reset_on_extrinsic:
                env_ = ResetOnExtrinsicReward(env_)
            return env_

        env = gym.vector.AsyncVectorEnv([
             lambda: _create_env(args) for _ in range(n_envs)
        ])
        env.action_space = env.action_space[0]
        normalize_rew = hyperparameters.get('normalize_rew', True)
        env = VecDontExtractDictObs(env, n_envs)
        if normalize_rew:
            env = VecNormalize(env, ob=False) # normalizing returns, but not the img frames.
        env = AtariNormFrame(env)
        env = TransposeFrame(env)
        env = ScaledFloatFrame(env)
        
    print(env_name, env.observation_space, env.action_space)
    ############
    ## LOGGER ##
    ############
    print('INITIALIZAING LOGGER...')
    logdir = 'procgen/' + env_name + '/' + exp_name + '/' + 'seed' + '_' + \
             str(seed) + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('logs', logdir)
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)
    logger = Logger(n_envs, logdir)

    ###########
    ## MODEL ##
    ###########
    print('INTIALIZING MODEL...')
    observation_space = env.observation_space
    observation_shape = observation_space.shape
    architecture = hyperparameters.get('architecture', 'impala')
    in_channels = observation_shape[0]
    action_space = env.action_space

    # Model architecture
    if architecture == 'nature':
        model = NatureModel(in_channels=in_channels)
    elif architecture == 'impala':
        model = ImpalaModel(in_channels=in_channels)
    elif architecture == 'simple':
        model = ActorCritic(in_channels, action_space.n)

    # Discrete action space
    recurrent = hyperparameters.get('recurrent', False)
    if isinstance(action_space, gym.spaces.Discrete):
        action_size = action_space.n
        policy = CategoricalPolicy(model, recurrent, action_size)
    else:
        raise NotImplementedError
    policy.to(device)

    #############
    ## STORAGE ##
    #############
    print('INITIALIZAING STORAGE...')
    hidden_state_dim = model.output_dim
    storage = Storage(observation_shape, hidden_state_dim, n_steps, n_envs, device)

    ###########
    ## AGENT ##
    ###########
    print('INTIALIZING AGENT...')
    algo = hyperparameters.get('algo', 'ppo')
    if algo == 'ppo':
        from agents.ppo import PPO as AGENT
    else:
        raise NotImplementedError
    agent = AGENT(env, policy, logger, storage, device, num_checkpoints, loss=loss, beta=beta, exploration_config=exploration_config, **hyperparameters)

    ##############
    ## TRAINING ##
    ##############
    print('START TRAINING...')
    agent.train(num_timesteps)
