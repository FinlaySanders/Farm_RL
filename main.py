from envs.farm_env import Farm_Env
from game import Game

env = Farm_Env(size=10)

#ppo = PPO(CNNPolicyNetwork, env, None)
#ppo.learn(total_timesteps=1000000)

game = Game()
game.run()








