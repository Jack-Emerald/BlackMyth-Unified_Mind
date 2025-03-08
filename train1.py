from stable_baselines3 import PPO
import os
from blackMyth_env import BlackMythEnv


def train(CREATE_NEW_MODEL, config):
	print("*** Training will begin shortly... Initializing...")

	TIMESTEPS = 1			#Learning rate multiplier.
	HORIZON_WINDOW = 2000	#Learning rate number of steps before updating the model. ~6min

	# Dynamically set model_name based on the boss number
	boss_number = config["BOSS"]  # Default to 1 if BOSS key is not present
	model_name = f"PPO-{boss_number}"

	'''Creating folder structure'''
	if not os.path.exists(f"models/{model_name}/"):
		os.makedirs(f"models/{model_name}/")
	if not os.path.exists(f"logs/{model_name}/"):
		os.makedirs(f"logs/{model_name}/")

	models_dir = f"models/{model_name}/"
	logdir = f"logs/{model_name}/"
	model_path = f"{models_dir}/PPO-1"
	print("*** Folders (models & logs) created...")


	'''Initializing environment'''
	env = BlackMythEnv(config)
	print("*** BlackMythEnv initialized...")


	'''Creating new model or loading existing model'''
	if CREATE_NEW_MODEL:
		model = PPO('MultiInputPolicy',
							env,
							tensorboard_log=logdir,
							n_steps=HORIZON_WINDOW,
							verbose=1,
							device='cpu')	#Set training device here.
		print("*** New Model created...")
	else:
		model = PPO.load(model_path, env=env)
		print("*** Model loaded...")


	'''Training loop'''
	while True:
		model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO", log_interval=1)
		model.save(f"{models_dir}/PPO-1")
		print(f"*** Model updated...")
