import gym

import math
import random
import numpy as np
from collections import namedtuple
from itertools import count
from PIL import Image
import cv2
import pickle
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import DQN
import Googles_DQN
from plots import *
from replay_memory import ReplayMemory


peregrine = True
load = False
train = True
use_negative_rewards = True
i_episode = 0
gamma = 0.9
seed = random.randint(0, 100)
start_learning = 15000
BATCH_SIZE = 100
GAMMA = 0.999
EPS_START = 1
EPS_END = 0.1
EPS_DECAY = 2000
TARGET_UPDATE = 10

episode_durations = []
cumulative_reward = []
running_loss = []


# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
is_ipython = True
if is_ipython:
    from IPython import display
    print("Ipython")

print("Making Breakout")
# env = gym.make('Breakout-v0').unwrapped
env = gym.make('BreakoutDeterministic-v4').unwrapped
env.reset()
img = plt.imshow(env.render(mode='rgb_array'))

print("Breakout Rendered")

plt.ion()

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'terminal'))


resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])


def get_screen():

    screen = env.render(mode='rgb_array')
    _, screen_height, screen_width = screen.shape

    screen = cv2.resize(screen, dsize=(80, 90))
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
    # Crop screen to show only relevant area to NN
    x = 0
    y = 10
    h = 90
    w = 80
    ######################################################
    # DEBUGGING
    ######################################################
    if not peregrine:
        cv2.namedWindow('Gray image', cv2.WINDOW_NORMAL)
        cv2.imshow('Gray image', screen[y:y+h, x:x+w])
        cv2.resizeWindow('Gray image', 400, 400)
        cv2.waitKey(1)

    screen = screen[y:y+h, x:x+w]
    screen = torch.from_numpy(screen)

    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).to(device)

init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape

# Get number of actions from gym action space
n_actions = env.action_space.n

policy_net = Googles_DQN.DQN(screen_height, screen_width, n_actions).to(device)
policy_net.apply(Googles_DQN.DQN.weights_init_uniform_rule)
target_net = Googles_DQN.DQN(screen_height, screen_width, n_actions).to(device)
print(target_net)
# GOOGLE: Initiliaze replay memory D to capacity N
memory = ReplayMemory(100000)

if load:
    policy_net.load_state_dict(torch.load('./modelcomplete.pyt', map_location=torch.device('cpu')))
    # with open('./modelMemory.pkl', 'rb') as pickle_file:
    #     memory = pickle.load(pickle_file)
    # with open('./cumulative_rewards.pkl', 'rb') as pickle_file:
    #     cumulative_reward = pickle.load(pickle_file)
    # with open('./episode_durations.pkl', 'rb') as pickle_file:
    #     episode_durations = pickle.load(pickle_file)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters(), lr=0.00001)


steps_done = 0


def select_action(state):
    global steps_done
    if steps_done > 1 and torch.cat(Transition(*zip(memory.memory[-1])).reward)[0] == -1.0:
        return torch.tensor([[1]], device=device, dtype=torch.long)
    sample = random.random()
    eps_threshold = 1
    if steps_done > start_learning:
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * (steps_done-start_learning) / EPS_DECAY)
    # print(steps_done)
    # print(eps_threshold)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


def optimize_model():
    if len(memory) < BATCH_SIZE or len(memory) < start_learning:
        return
    # GOOGLE: Line 11
    for s in range(BATCH_SIZE):
        index = random.randint(4, len(memory))
        t = memory.sample(index)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    # for t in transitions:
        batch = Transition(*zip(*t))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        for idx, r in enumerate(reward_batch):
            if r == 1:
                # https://github.com/fg91/Deep-Q-Learning/blob/master/DQN.ipynb
                # Reward at t=0 is associated with the  reward state,
                # We take the states leading up to this
                t = memory.sample(index-(4+idx))
                batch = Transition(*zip(*t))

                # Compute a mask of non-final states and concatenate the batch elements
                # (a final state would've been the one after which simulation ended)
                non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                        batch.next_state)), device=device, dtype=torch.bool)
                non_final_next_states = torch.cat([s for s in batch.next_state
                                                   if s is not None])
                state_batch = torch.cat(batch.state)
                action_batch = torch.cat(batch.action)
                reward_batch = torch.cat(batch.reward)

                reward_batch[-1] = 1
                for i in range(0, len(reward_batch[:-1])):
                    reward_batch[-2 - i] = reward_batch[-1 - i] * (gamma ** (i + 1))
                break
            if r == -1:
                # https://github.com/fg91/Deep-Q-Learning/blob/master/DQN.ipynb
                # Reward at t=0 is associated with the  reward state,
                # We take the states leading up to this
                t = memory.sample(index-(4+idx))
                batch = Transition(*zip(*t))

                # Compute a mask of non-final states and concatenate the batch elements
                # (a final state would've been the one after which simulation ended)
                non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                        batch.next_state)), device=device, dtype=torch.bool)
                non_final_next_states = torch.cat([s for s in batch.next_state
                                                   if s is not None])
                state_batch = torch.cat(batch.state)
                action_batch = torch.cat(batch.action)
                reward_batch = torch.cat(batch.reward)

                reward_batch[-1] = -1
                for i in range(0, len(reward_batch[:-1])):
                    reward_batch[-2-i] = reward_batch[-1-i]*(gamma**(i+1))
                break
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(len(t), device=device)
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        # Q = r + gamma*max Q'
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        running_loss[-1] += loss.data
        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()


# GOOGLE: M
num_episodes = 300
# GOOGLE: for episode =1 do
for i_episode in range(num_episodes):
    print("Training Episode %s" % i_episode)
    # Initialize the environment and state
    temp_reward = 0
    running_loss.append(0)
    env.reset()
    last_lives = 0

    last_screen = get_screen()
    current_screen = get_screen()
    # Why is state = the difference?
    state = current_screen - last_screen
    for t in count():
        # GOOGLE: lines 6+7
        action = select_action(state)

        # GOOGLE: line 8
        _, reward, done, info = env.step(action.item())
        temp_reward += reward


        if use_negative_rewards:
            if info['ale.lives'] < last_lives:
                terminal_life_lost = True
                reward = -1.0
            else:
                terminal_life_lost = done
        else:
            terminal_life_lost = done
        last_lives = info['ale.lives']

        reward = torch.tensor([reward], device=device)

        # Observe new state
        last_screen = current_screen
        last_action = action
        current_screen = get_screen()
        # END line 8

        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None
        # GOOGLE: line 10
        # Store the transition in memory
        memory.push(state, action, next_state, reward, terminal_life_lost)

        # Move to the next state
        state = next_state
        # Perform one step of the optimization (on the target network)
        env.render()
        if train:
            optimize_model()

        if done:
            cumulative_reward.append(temp_reward)
            episode_durations.append(t + 1)
            if i_episode % 10 == 0:
                plot_durations(is_ipython, episode_durations, seed, save_fig=True)
                plot_rewards(is_ipython, cumulative_reward, seed, save_fig=True)
                plot_loss(is_ipython, running_loss, seed, save_fig=False)
            break
    if i_episode % 100 == 0:
        torch.save(policy_net.state_dict(), ('./model/model' + str(i_episode)))
        with open('./model/running_memory.pkl', 'wb') as output:
            pickle.dump(memory, output, pickle.HIGHEST_PROTOCOL)
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
    i_episode += 1
print('Complete')
print(i_episode)
plot_durations(is_ipython, episode_durations, seed, save_fig=True)
plot_rewards(is_ipython, cumulative_reward, seed, save_fig=True)

torch.save(policy_net.state_dict(), './model/modelcomplete.pyt')
with open('./model/modelMemory'+str(seed) + '.pkl', 'wb') as output:
    pickle.dump(memory, output, pickle.HIGHEST_PROTOCOL)
with open('./model/cumulative_rewards'+str(seed) + '.pkl', 'wb') as output:
    pickle.dump(cumulative_reward, output, pickle.HIGHEST_PROTOCOL)
with open('./model/episode_durations'+str(seed) + '.pkl', 'wb') as output:
    pickle.dump(episode_durations, output, pickle.HIGHEST_PROTOCOL)
env.close()
plt.ioff()


