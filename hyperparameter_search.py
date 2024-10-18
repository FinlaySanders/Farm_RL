from envs.farm_env import Farm_Env
from game import Game
import wandb

from PPO_Implemetations.cnn_ppo import PPO
from policynets import CNNPolicyNetwork
from world_editor import World_Generator


sweep_config = {
    'method': 'bayes'
}

metric = {
    'name': 'Average Episodic Return',
    'goal': 'maximize'   
    }
sweep_config['metric'] = metric

parameters_dict = {
    'timesteps_per_batch': {
        'distribution': 'int_uniform',
        'min': 2000,
        'max': 10000
    },
    'max_timesteps_per_episode': {
        'distribution': 'int_uniform',
        'min': 500,
        'max': 1500
    },
    'n_updates_per_iteration': {
        'distribution': 'int_uniform',
        'min': 1,
        'max': 10
    },
    'lr': {
        'distribution': 'log_uniform_values',
        'min': 1e-5,
        'max': 1e-2
    },
    'gamma': {
        'distribution': 'uniform',
        'min': 0.9,
        'max': 0.99
    },
    'clip': {
        'distribution': 'uniform',
        'min': 0.1,
        'max': 0.3
    },
    'lam': {
        'distribution': 'uniform',
        'min': 0.9,
        'max': 0.99
    },
    'num_minibatches': {
        'distribution': 'int_uniform',
        'min': 4,
        'max': 8
    },
    'ent_coef': {
        'distribution': 'uniform',
        'min': 0.0,
        'max': 0.05
    },
    'target_kl': {
        'distribution': 'uniform',
        'min': 0.01,
        'max': 0.05
    },
    'max_grad_norm': {
        'distribution': 'uniform',
        'min': 0.1,
        'max': 1.0
    },
}
sweep_config['parameters'] = parameters_dict

sweep_id = wandb.sweep(sweep_config, project="farm-RL")

def train(config=None):
    with wandb.init(config=config):
        config = wandb.config

        world_size = 10
        world_generator = World_Generator(world_size)
        env = Farm_Env(world_size, world_generator)

        ppo = PPO(CNNPolicyNetwork, env, wandb.config, wandb_log=True, save_models=False)
        ppo.learn(total_timesteps=300000)

wandb.agent(sweep_id, train, count=200)

