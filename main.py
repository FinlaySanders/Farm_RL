from PPO_Implemetations.cnn_ppo import PPO
from envs.farm_env import Farm_Env
from policynets import CNNPolicyNetwork

from game import Game

env = Farm_Env(size=10)

#ppo = PPO(CNNPolicyNetwork, env, None)
#ppo.learn(total_timesteps=1000000)

game = Game()
game.run()








