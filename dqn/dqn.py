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
                new_state,reward,terminated, truncared, _=env.step(action)

                #save exp in memory
                memory.append((state, action, new_state, reward, terminated))

                #next state
                state=new_state

                #increment step count
                step+=1

            #keep track of rewards collected
            if reward==1:
                rewards_per_ep[i]=1

            #check if enough exp has been collected and atleast 1 reward got
            if len(memory)>self.min_batch_size and np.sum(rewards_per_ep)>0:
                min_batch=memory.sample(self.min_batch_size)
                self.optimize(min_batch, policy_dqn,target_dqn)

                #decay the epsilon
                epsilon=max(epsilon-1/episodes,0)
                epsilon_history.append(epsilon)

                #copy pol to target after some number of steps
                if step>self.newtork_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step=0
        #close env
        env.close()

        #save pol
        torch.save(policy_dqn.state_dict(), 'frozenlake.pt')

        #create new graph
        plt.figure(1)

         # Plot average rewards (Y-axis) vs episodes (X-axis)
        sum_rewards = np.zeros(episodes)
        for x in range(episodes):
            sum_rewards[x] = np.sum(rewards_per_episode[max(0, x-100):(x+1)])
        plt.subplot(121) # plot on a 1 row x 2 col grid, at cell 1
        plt.plot(sum_rewards)
        
        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122) # plot on a 1 row x 2 col grid, at cell 2
        plt.plot(epsilon_history)
        
        # Save plots
        plt.savefig('frozen_lake_dql.png')

    # Optimize policy network
    def optimize(self, mini_batch, policy_dqn, target_dqn):

        # Get number of input nodes
        num_states = policy_dqn.fc1.in_features

        current_q_list = []
        target_q_list = []

        for state, action, new_state, reward, terminated in mini_batch:

            if terminated: 
                # Agent either reached goal (reward=1) or fell into hole (reward=0)
                # When in a terminated state, target q value should be set to the reward.
                target = torch.FloatTensor([reward])
            else:
                # Calculate target q value 
                with torch.no_grad():
                    target = torch.FloatTensor(
                        reward + self.discount_factor_g * target_dqn(self.state_to_dqn_input(new_state, num_states)).max()
                    )

            # Get the current set of Q values
            current_q = policy_dqn(self.state_to_dqn_input(state, num_states))
            current_q_list.append(current_q)

            # Get the target set of Q values
            target_q = target_dqn(self.state_to_dqn_input(state, num_states)) 
            # Adjust the specific action to the target that was just calculated
            target_q[action] = target
            target_q_list.append(target_q)
                
        # Compute loss for the whole minibatch
        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    '''
    Converts an state (int) to a tensor representation.
    For example, the FrozenLake 4x4 map has 4x4=16 states numbered from 0 to 15. 

    Parameters: state=1, num_states=16
    Return: tensor([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    '''
    def state_to_dqn_input(self, state:int, num_states:int)->torch.Tensor:
        input_tensor = torch.zeros(num_states)
        input_tensor[state] = 1
        return input_tensor

    # Run the FrozeLake environment with the learned policy
    def test(self, episodes, is_slippery=False):
        # Create FrozenLake instance
        env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=is_slippery, render_mode='human')
        num_states = env.observation_space.n
        num_actions = env.action_space.n

        # Load learned policy
        policy_dqn = DQN(in_states=num_states, h1_nodes=num_states, out_actions=num_actions) 
        policy_dqn.load_state_dict(torch.load("frozen_lake_dql.pt"))
        policy_dqn.eval()    # switch model to evaluation mode

        print('Policy (trained):')
        self.print_dqn(policy_dqn)

        for i in range(episodes):
            state = env.reset()[0]  # Initialize to state 0
            terminated = False      # True when agent falls in hole or reached goal
            truncated = False       # True when agent takes more than 200 actions            

            # Agent navigates map until it falls into a hole (terminated), reaches goal (terminated), or has taken 200 actions (truncated).
            while(not terminated and not truncated):  
                # Select best action   
                with torch.no_grad():
                    action = policy_dqn(self.state_to_dqn_input(state, num_states)).argmax().item()

                # Execute action
                state,reward,terminated,truncated,_ = env.step(action)

        env.close()

    # Print DQN: state, best action, q values
    def print_dqn(self, dqn):
        # Get number of input nodes
        num_states = dqn.fc1.in_features

        # Loop each state and print policy to console
        for s in range(num_states):
            #  Format q values for printing
            q_values = ''
            for q in dqn(self.state_to_dqn_input(s, num_states)).tolist():
                q_values += "{:+.2f}".format(q)+' '  # Concatenate q values, format to 2 decimals
            q_values=q_values.rstrip()              # Remove space at the end

            # Map the best action to L D R U
            best_action = self.ACTIONS[dqn(self.state_to_dqn_input(s, num_states)).argmax()]

            # Print policy in the format of: state, action, q values
            # The printed layout matches the FrozenLake map.
            print(f'{s:02},{best_action},[{q_values}]', end=' ')         
            if (s+1)%4==0:
                print() # Print a newline every 4 states

if __name__ == '__main__':

    frozen_lake = FrozenLakeDQL()
    is_slippery = False
    frozen_lake.train(1000, is_slippery=is_slippery)
    frozen_lake.test(10, is_slippery=is_slippery)