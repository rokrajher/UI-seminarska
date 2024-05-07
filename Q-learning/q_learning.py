import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

def run(episodes, is_training, render, is_slippery, map_name, learning_rate_a, discount_factor_g, epsilon, epsilon_decay_rate, seed):
    size = map_name.split('x')[0]
    size = int(size)
    alpha = learning_rate_a
    gamma = discount_factor_g

    env = gym.make('FrozenLake-v1', desc=generate_random_map(size=size, seed=seed), is_slippery=is_slippery, render_mode='human' if render else None)

    if(is_training):
        q = np.zeros((env.observation_space.n, env.action_space.n)) # init a 64 x 4 array
    else:
        f = open(f'./models/frozen_lake_{map_name}_{learning_rate_a}_{discount_factor_g}_{is_slippery}.pkl', 'rb')
        q = pickle.load(f)
        f.close()

    rng = np.random.default_rng(seed=seed)   # random number generator

    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0]  # states: 0 to 63, 0=top left corner,63=bottom right corner
        terminated = False      # True when fall in hole or reached goal
        truncated = False       # True when actions > 200

        while(not terminated and not truncated):
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample() # actions: 0=left,1=down,2=right,3=up
            else:
                action = np.argmax(q[state,:])

            new_state,reward,terminated,truncated,_ = env.step(action)

            if is_training:
                q[state,action] = q[state,action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state,:]) - q[state,action]
                )

            state = new_state

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        if(epsilon==0):
            learning_rate_a = 0.0001

        if reward == 1:
            rewards_per_episode[i] = 1

    env.close()

    if is_training:
        f = open(f'./models/frozen_lake_{map_name}_alpha_{alpha}_gamma_{gamma}_slippery_{is_slippery}.pkl', 'wb')
        pickle.dump(q, f)
        f.close()
        
    return rewards_per_episode

if __name__ == '__main__':
    reward_per_episode = run(episodes=15000, is_training=False, is_slippery=True,  render=False, map_name="8x8", learning_rate_a=0.9, discount_factor_g=0.9, epsilon=1, epsilon_decay_rate=0.0001, seed=10)