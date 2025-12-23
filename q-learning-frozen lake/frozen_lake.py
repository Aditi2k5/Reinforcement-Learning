import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def run(episodes, is_training=True, render=False):
    env=gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True, render_mode='human' if render else None)
    
    if (is_training):
        q=np.zeros((env.observation_space.n, env.action_space.n)) #init the array
    else:
        f=open('frozen_lakes.pkl','rb')
        q=pickle.load(f)
        f.close()


    learning_rate_a=0.9 #alpha for q learning
    discount_factor_g=0.9 #gamma

    #epsilon greedy approach
    epsilon=1 #1=fully random actions, ovvertime can decrease randomness
    epsilon_decay_rate=0.0001 #decay rate, direct impact on no of eps needed to train
    # for it to go till zero, train for 10k eps as 1/0.0001
    rng=np.random.default_rng() #rando num generator
    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state=env.reset()[0]
        terminated=False
        truncated=False

        while(not terminated and not truncated):
            if is_training and rng.random()<epsilon:
                action = env.action_space.sample() #take random actions, 0=left, 1=down, 2=right, 3=up
            else:
                action=np.argmax(q[state,:]) # follow q table
            new_state, reward, terminated, truncated,_=env.step(action)

            if is_training:
                q[state, action]=q[state, action] + learning_rate_a *(
                    reward+discount_factor_g * np.max(q[new_state,:])-q[state,action])

            state=new_state
    
        epsilon = max(epsilon-epsilon_decay_rate, 0) #after each episode decrease epsilon all the way till zero

        if (epsilon==0):
            learning_rate_a=0.0001 #reduce lr to help stabilize q values once we r not exploring much

        if reward==1:
            rewards_per_episode[i]=1
    env.close()

    sum_rewards = np.zeros(episodes)
    for j in range(episodes):
        sum_rewards[j] = np.sum(rewards_per_episode[max(0,j-100):(j+1)])
    plt.plot(sum_rewards)
    plt.savefig('frozen_lakes_new.png')

    if is_training:
        f=open('frozen_lakes.pkl','wb')
        pickle.dump(q,f)
        f.close()

if __name__ == '__main__':
    run(1000, is_training=False, render=False)