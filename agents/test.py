from stable_baselines3 import SAC

model = SAC.load("/Users/jaysm/Desktop/UTS_FILES/Codes/Sem4_Spring26/RL/Assign3/donkey-rl-scaffold/checkpoints/final_models/sac_donkey_250000_steps.zip")
print(model.observation_space)
print(model.observation_space.shape)