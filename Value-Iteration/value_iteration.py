import argparse
import pickle
import gym
import numpy as np
# import tools
from matplotlib import pyplot as plt
from gym.envs.toy_text.frozen_lake import generate_random_map



# def argmax(env, V, pi, action,s, gamma):
#     e = np.zeros(env.action_space.n)
#     for a in range(env.action_space.n):                         # iterate for every action possible 
#         q=0
#         P = np.array(env.env.P[s][a])                   
#         (x,y) = np.shape(P)                             # for Bellman Equation 
        
#         for i in range(x):                              # iterate for every possible states
#             s_= int(P[i][1])                            # S' - Sprime - possible succesor states
#             p = P[i][0]                                 # Transition Probability P(s'|s,a) 
#             r = P[i][2]                                 # Reward
            
#             q += p*(r+gamma*V[s_])                      # calculate action_ value q(s|a)
#             e[a] = q
            
#     m = np.argmax(e) 
#     action[s]=m                                           # Take index which has maximum value 
#     pi[s][m] = 1                                        # update pi(a|s) 

#     return pi


# def bellman_optimality_update(env, V, s, gamma):  # update the stae_value V[s] by taking 
#     pi = np.zeros((env.observation_space.n, env.action_space.n))       # action which maximizes current value
#     e = np.zeros(env.action_space.n)                       
#                                             # STEP1: Find 
#     for a in range(env.action_space.n):             
#         q=0                                 # iterate for all possible action
#         P = np.array(env.env.P[s][a])
#         (x,y) = np.shape(P)
        
#         for i in range(x):
#             s_= int(P[i][1])
#             p = P[i][0]
#             r = P[i][2]
#             q += p*(r+gamma*V[s_])
#             e[a] = q
            
#     m = np.argmax(e)
#     pi[s][m] = 1
    
#     value = 0
#     for a in range(env.action_space.n):
#         u = 0
#         P = np.array(env.env.P[s][a])
#         (x,y) = np.shape(P)
#         for i in range(x):
            
#             s_= int(P[i][1])
#             p = P[i][0]
#             r = P[i][2]
            
#             u += p*(r+gamma*V[s_])
            
#         value += pi[s,a] * u
  
#     V[s]=value
#     return V[s]


# def value_iteration(env, gamma, theta):
#     V = np.zeros(env.observation_space.n)                                       # initialize v(0) to arbitory value, my case "zeros"
#     while True:
#         delta = 0
#         for s in range(env.observation_space.n):                       # iterate for all states
#             v = V[s]
#             bellman_optimality_update(env, V, s, gamma)   # update state_value with bellman_optimality_update
#             delta = max(delta, abs(v - V[s]))             # assign the change in value per iteration to delta  
#         if delta < theta:                                       
#             break                                         # if change gets to negligible 
#                                                           # --> converged to optimal value         
#     pi = np.zeros((env.observation_space.n, env.action_space.n)) 
#     action = np.zeros((env.observation_space.n))
#     for s in range(env.observation_space.n):
#         pi = argmax(env, V, pi,action, s, gamma)         # extract optimal policy using action value 
        
#     return V, pi,action                                          # optimal value funtion, optimal policy

def value_iteration(env, num_iterations=1000, threshold=1e-20, gamma=1.0, value_table=None):

    #initialize the value table
    if value_table is None:
        value_table = np.zeros(env.observation_space.n)
    
    for i in range(num_iterations):
        updated_value_table = np.copy(value_table) 
     
        #compute the Q values of all the actions in the state
        for s in range(env.observation_space.n):
            Q_values = [sum([prob*(r + gamma * updated_value_table[s_])
                             for prob, s_, r, _ in env.P[s][a]]) 
                                   for a in range(env.action_space.n)] 
                                        
            value_table[s] = max(Q_values) # take max value
                        
        # check if we have reached the convergence i.e the difference between our value table
        if (np.sum(np.fabs(updated_value_table - value_table)) <= threshold):
            return value_table, i, (np.sum(np.fabs(updated_value_table - value_table)))
    
    return value_table, num_iterations, (np.sum(np.fabs(updated_value_table - value_table)))

def extract_policy(env, value_table, gamma=1.0):
    
    #initialize the policy with zeros
    policy = np.zeros(env.observation_space.n) 
    policy_values = np.zeros(env.observation_space.n)
    
    #for each state
    for s in range(env.observation_space.n):
        
        #compute the Q value of all the actions in the state
        Q_values = [sum([prob*(r + gamma * value_table[s_])
                             for prob, s_, r, _ in env.P[s][a]]) 
                                   for a in range(env.action_space.n)] 
                
        #extract policy by selecting the action which has maximum Q value
        policy[s] = np.argmax(np.array(Q_values))
        policy_values[s] = max(Q_values) 
    
    return policy, policy_values

def run(render, env_size, is_slippery, n_iterations, gamma, theta, seed):
    size = env_size
    gamma = gamma

    env = gym.make('FrozenLake-v1', desc=generate_random_map(size=size), is_slippery=is_slippery, render_mode='human' if render else None)

    V, steps, value = value_iteration(env, num_iterations=n_iterations, gamma=gamma, threshold=theta)
    pi, pi_values = extract_policy(env, V, gamma=gamma)
    reward, rewards = evaluate_policy(env, pi, num_episodes=100)

    env.close()

    return V, pi, pi_values, steps, value, reward

def policy_run(render, env_size, is_slippery, n_iterations, gamma, theta, seed):
    size = env_size
    gamma = gamma

    env = gym.make('FrozenLake-v1', desc=generate_random_map(size=size), is_slippery=is_slippery, render_mode='human' if render else None)
    V = None
    reward = 0
    rewards = [0.0]
    all_rewards_per_episode = []
    all_successful_episodes = []
    iterations_to_converge = n_iterations
    for i in range(n_iterations):
        V, steps, value = value_iteration(env, num_iterations=1, gamma=gamma, threshold=theta, value_table=V)
        pi, pi_values = extract_policy(env, V, gamma=gamma)
        reward, total_rewards, successful_episodes = evaluate_policy(env, pi, num_episodes=100)
        

        # all_rewards_per_episode.append(rewards_per_episode)
        all_successful_episodes.append(successful_episodes)
        rewards.append(reward)

        if value < theta:
            iterations_to_converge = i
            #return V, pi, reward, rewards,all_successful_episodes, i

    env.close()

    return V, pi, reward, rewards, all_successful_episodes, iterations_to_converge

def evaluate_policy(env, policy, num_episodes=100):
    total_rewards = []
    rewards_per_episode = []
    successfull_episodes = 0

    for _ in range(num_episodes):
        state = env.reset()[0]
        done = False
        total_reward = 0
        episode_rewards = []
        steps = 0
        while not done and steps < num_episodes:
            action = int(policy[state])
            state, reward, done, _, prob = env.step(action)
            total_reward += reward
            episode_rewards.append(reward)
            steps += 1
            
        # rewards_per_episode.append(episode_rewards)
        total_rewards.append(total_reward)
        if total_reward == 1.0:
            successfull_episodes += 1


    average_reward = np.mean(total_rewards)
    return average_reward, total_rewards, successfull_episodes #rewards_per_episode, successfull_episodes

def get_ema(rewards, smoothing_factor=0.9):
    ema = [rewards[0]]
    for r in rewards[1:]:
        ema.append(smoothing_factor * ema[-1] + (1 - smoothing_factor) * r)
    return ema

if __name__ == '__main__':

    args = argparse.ArgumentParser()
    args.add_argument("--env_size", type=int, default=4, choices=[4,8,16,32])
    args.add_argument("--gamma", default=0.99)
    args.add_argument("--theta", default=0.000001)
    args.add_argument("--seed", default=10)
    args.add_argument("--is_slippery", default=True)
    args.add_argument("--render", default=False)
    args.add_argument("--iterations", default=1000)
    args = args.parse_args()

    V, pi, reward, rewards, rewards_per_episode, i = policy_run(render=False, is_slippery=True, n_iterations=30, env_size=4, seed=20, gamma=0.8, theta=0.01)


    V, pi, pi_values, steps, value = run(args.render, args.env_size, args.is_slippery, args.iterations, args.gamma, args.theta, args.seed)
    print('Value Iteration converged in {} steps with value {}'.format(steps, value))


