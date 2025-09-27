import gymnasium as gym
import numpy as np
import pandas as pd
# import torch
from mpc_ilqr import CartPoleILQR
# from learn import ImitationNet

########## Controller selection ##########
controller = "moc" # opt : "mpc", "il"
num_episodes = 100
##########################################

if controller == "mpc":
    mpc = CartPoleILQR(horizon=40, dt=0.02)
    env = gym.make('CartPole-v1', render_mode='rgb_array')
# elif controller == "il":
#     il_model = ImitationNet()
#     il_model.load_state_dict(torch.load('cartpole_imitation.pth'))
#     il_model.eval()
#     env = gym.make('CartPole-v1', render_mode='human')
else:
    raise ValueError("Unknown controller type")


dataset = []
suc = 0

for i_episode in range(num_episodes):
    observation, info = env.reset()
    temp = []

    for t in range(500):
        old = observation.copy()
        # if controller == "il":
        #     obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        #     with torch.no_grad():
        #         raw_action = il_model(obs_tensor)
            
        if controller == "mpc":
            raw_action = mpc.control(np.array(observation))
            

        # Gym action (discrete)
        action = int(raw_action > 0)
        observation, reward, terminated, truncated, info = env.step(action)

        temp.append([*old, raw_action, action])

        if terminated or truncated:
            print(f"Episode {i_episode} finished after {t+1} timesteps")
            break
    
    if t+1 >= 400:
        suc += 1
        dataset.append(temp)
    
print(f"Success episode {suc}")

if controller == "mpc":
    # Save dataset
    flat_dataset = [row for ep in dataset for row in ep]
    df = pd.DataFrame(flat_dataset, columns=['x','x_dot','theta','theta_dot','raw_action','discrete_action'])
    df.to_csv('cartpole_mpc_data.csv', index=False)
elif controller == "il":
    pass

env.close()
