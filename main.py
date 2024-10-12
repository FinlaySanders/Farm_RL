from envs.farm_env import Farm_Env
from game import Game

from PPO_Implemetations.cnn_ppo import PPO
from policynets import CNNPolicyNetwork
from world_editor import World_Generator


world_size = 10
world_generator = World_Generator(world_size)
env = Farm_Env(world_size, world_generator)

#ppo = PPO(CNNPolicyNetwork, env, None)
#ppo.learn(total_timesteps=1000000)

game = Game(env, "ppo_actor_600509.pth")
game.run()






