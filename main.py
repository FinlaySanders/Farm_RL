from envs.farm_env import Farm_Env
from game import Game
import wandb

from PPO_Implemetations.cnn_ppo import PPO
from policynets import CNNPolicyNetwork
from world_editor import World_Generator

hyperparameters = {
    "timesteps_per_batch": 4500,                  # Number of timesteps to run per batch
    "max_timesteps_per_episode": 1100,            # Max number of timesteps per episode
    "n_updates_per_iteration": 8,                 # Number of times to update actor/critic per iteration
    "lr": 0.0035,                                 # Learning rate of actor optimizer
    "gamma": 0.9,                                 # Discount factor to be applied when calculating Rewards-To-Go
    "clip": 0.195,                                # Recommended 0.2, helps define the threshold to clip the ratio during SGA
    "lam": 0.92,                                  # Lambda Parameter for GAE 
    "num_minibatches": 7,                         # Number of mini-batches for Mini-batch Update
    "ent_coef": 0.045,                            # Entropy coefficient for Entropy Regularization
    "target_kl": 0.046,                           # KL Divergence threshold
    "max_grad_norm": 0.16,                        # Gradient Clipping threshold
}


#wandb.init(
#    project="farm-RL",
#    config=hyperparameters
#)

world_size = 10
world_generator = World_Generator(world_size)
env = Farm_Env(world_size, world_generator)

#ppo = PPO(CNNPolicyNetwork, env, hyperparameters, wandb_log=True, save_models=True)
#ppo.learn(total_timesteps=1000000)

game = Game(env, "ppo_actor_151442.pth")
game.run()






