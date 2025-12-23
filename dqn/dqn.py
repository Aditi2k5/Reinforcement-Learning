import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque
import torch
from torch import nn
import torch.nn.functional as nn
import torch.nn.functional as F
#define model
class dqn(nn.Module):
    def __init__(self, in_states,h1_nodes, out_actions):
        super().__init__()

        #define the layers
        self.fc1 = nn.Linear(in_states,h1_nodes) #first full connected layer
        self.out=nn.Linear(h1_nodes, out_actions) # output layer

    def forward(self, x):
        x=F.relu(self.fc1(x)) #apply relu activation
        x=self.out(x) #calc op
        return x

#define the memory for exp replay
class Replay():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)

    def append(self, transition): # transition is the tuple (s,a,new_state,re, terminated)
        self.memory.append(transition)
    
    def sample(self, sample_size): #return a random sample of wtv size we want from memory
        return random.sample(self.memory, sample_size)
    def __len__(self):
        return len(self.memory) #return len of memory

#FrozenLake Deep Qlearning

class FrozenLakeDQL():
    #hyperparams
    learning_rate_a = 0.01 #learning rate alpha
    discount_factor_g=0.9 #discount rate gamma
    newtork_sync_rate=10 #no of steps agen ttakes before syncing policy and target network 
    replay_memory_size=1000 # size of rpelay memory
    min_batch_size=32 #size of training dataset sampled from replay memory

    #neural net
    loss_fn=nn.MSELoss()
    optimizer=None #nn optimizer, init later

    Actions=['L','D','R','U'] #0,1,2,3 = L,D,R,U

    #train the environment
    def train(self, episodes, render=False, is_slippery=False):
        env=gym.make('FrozenLake-v1', map_name="4x4", is_slippery=is_slippery, render_mode='human' if render else None)
        num_states=env.observation_space.n
        num_actions=env.action_space.np_random

        epsilon=1 # fully random actions
        memory= Replay(self.replay_memory_size)


        #make a plicy and target network, no of nodes in hidden layer can be adjusted
        policy_dqn= dqn(in_states=num_states, h1_nodes=num_states, out_actions=num_actions)
        target_dqn= dqn(in_states=num_states, h1_nodes=num_states, out_actions=num_actions)

        #make them the same by copying the weights from one ntwk to other
        target_dqn.load_state_dict(policy_dqn.state_dict())

        print('Policy (random ,before training):')
        self.print_dqn(policy_dqn)

        #pol ntwk optimimzeer, adam for now can be swapped later
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

        #list to track rewards for each ep, first 0
        rewards_per_ep = np.zeros(episodes)

        #list to track ep decay
        ep_history=[]

        #track steps taken, needed for syncing policy to target
        step=0

        for i in range(episodes):
            state=env.reset()[0]
            terminated=False #true if agent falls or reached goal
            truncated=False #true if agent takes more than 200 actions

            while(not terminated and not truncated):
                if random.random()<epsilon:
                    action = env.action_space.sample() #take random actions, 0=left, 1=down, 2=right, 3=up
                else:
                    with torch.no_grad():
                        action = policy_dqn(self.state_to_dqn_input(state, num_states)).argmax().item()
                
                #exec action
                new_state,reward,terminared



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