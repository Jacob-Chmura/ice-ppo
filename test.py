import argparse
import gym
import numpy as np
import os
import random
import torch
import yaml

import imageio
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from common import set_global_seeds
from common.env.procgen_wrappers import AtariNormFrame
from common.env.procgen_wrappers import ScaledFloatFrame
from common.env.procgen_wrappers import TransposeFrame
from common.env.procgen_wrappers import VecDontExtractDictObs
from common.model import NatureModel, ImpalaModel, ActorCritic
from common.policy import CategoricalPolicy
from common.ice import ICEOptGPU as ICE


parser = argparse.ArgumentParser(
    description="Information Content Exploration Test Script"
)
parser.add_argument('--env_name',          type=str, default='MontezumaRevenge-v0',  help='environment ID')
parser.add_argument('--exp_name',         type=str, default = 'test_optimized_ice', help='experiment name')
parser.add_argument('--start_level',       type=int, default=0,                      help='start-level for environment')
parser.add_argument('--num_levels',        type=int, default=500,                    help='number of training levels for environment')
parser.add_argument('--distribution_mode', type=str, default='hard',                 help='distribution mode for environment')
parser.add_argument('--param_name',        type=str, default='hard',                 help='hyper-parameter ID')
parser.add_argument('--gpu_device',        type=int, default=0,                      help='visible device in CUDA')
parser.add_argument('--num-episodes',      type=int, default=10,                     help='number of testing episodes')
parser.add_argument('--seed',              type=int, default=random.randint(0,9999), help='Random generator seed')

ATARI_ACTION_MAPPING = {
    0: "NOOP",
    1: "FIRE",
    2: "UP",
    3: "RIGHT",
    4: "LEFT",
    5: "DOWN",
    6: "UP-RIGHT",
    7: "UP-LEFT",
    8: "DOWN-RIGHT",
    9: "DOWN-LEFT",
    10: "UP-FIRE",
    11: "RIGHT-FIRE",
    12: "LEFT-FIRE",
    13: "DOWN-FIRE",
    14: "UP-RIGHT-FIRE",
    15: "UP-LEFT-FIRE",
    16: "DOWN-RIGHT-FIRE",
    17: "DOWN-LEFT-FIRE",
}

if __name__=='__main__':
    args = parser.parse_args()
    set_global_seeds(args.seed)

    with open('hyperparams/procgen/config.yml', 'r') as f:
        hyperparameters = yaml.safe_load(f)[args.param_name]

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_device)
    device = torch.device('cuda')
    n_envs = 1
    torch.set_num_threads(1)
    env = gym.vector.AsyncVectorEnv([
            lambda: gym.make(args.env_name) for _ in range(n_envs)
    ])
    env.action_space = env.action_space[0]
    env = VecDontExtractDictObs(env, n_envs)
    env = AtariNormFrame(env)
    env = TransposeFrame(env)
    env = ScaledFloatFrame(env)
    print(args.env_name, env.observation_space, env.action_space)

    architecture = hyperparameters.get('architecture', 'impala')
    in_channels = env.observation_space.shape[0]
    action_space = env.action_space
    if architecture == 'nature':
        model = NatureModel(in_channels=in_channels)
    elif architecture == 'impala':
        model = ImpalaModel(in_channels=in_channels)
    elif architecture == 'simple':
        model = ActorCritic(in_channels, env.action_space.n)
    else:
        raise ValueError(f"Unrecognized architecture: {architecture}")

    # Discrete action space
    recurrent = hyperparameters.get('recurrent', False)
    policy = CategoricalPolicy(model, recurrent, env.action_space.n)

    # The whole logging file path needs to be cleaned up so we can easily grab the most recent model
    policy.load_state_dict(torch.load("logs/procgen/MontezumaRevenge-v0/test_optimized_ice/seed_6426_24-03-2023_09-55-07/model_175005696.pth")["state_dict"])
    policy.to(device)

    print('START TESTING...')
    num_episodes = 1
    ice = ICE(beta=50)

    for episode in range(num_episodes):
        policy.eval()
        obs = env.reset()
        ice.reset(obs)
        hidden_state = torch.zeros(size=(n_envs, model.output_dim), device=device)
        done = False
        step = 0
        frames, annotations = [], []
        while not done:
            obs = torch.FloatTensor(obs).to(device=device)
            mask = torch.FloatTensor(1-done).to(device=device)
            dist, _, hidden_state = policy(obs, hidden_state, mask)
            act = dist.sample()
            obs, rew_extrinsic, done, info = env.step(act)
            rew_intrinsic = ice.update(obs, done)
            frames.append(np.moveaxis(obs[0], 0, -1))

            annotations.append(f"Step {step} | {ATARI_ACTION_MAPPING[act.cpu().numpy()[0]]}\nInformation Gain: {rew_intrinsic[0]:3f}")
            step += 1

        ann_frames = []
        for i, frame in enumerate(frames):
            f, ax = plt.subplots(figsize=(12, 12), dpi=40)
            ax.imshow(frame)
            ax.text(5, 5, annotations[i], fontsize="xx-large", va="top", color="red")
            ax.set_axis_off()
            ax.set_position([0, 0, 1, 1])
            f.canvas.draw()
            ann_frames.append(np.asarray(f.canvas.renderer.buffer_rgba()))
            plt.close()
        imageio.mimsave(f"test_{episode}.gif", ann_frames, fps=5)
    env.close()
