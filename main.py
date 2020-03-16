import gym

import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import cv2
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import scipy.misc
from gym.wrappers import AtariPreprocessing

import DQN
import Googles_DQN
from replay_memory import ReplayMemory

load = False
train = True
i_episode = 0
gamma = 0.7
seed = random.randint(0,100)
start_learning = 1000
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
    cv2.namedWindow('Gray image', cv2.WINDOW_NORMAL)
    cv2.imshow('Gray image', screen[y:y+h, x:x+w])
    cv2.resizeWindow('Gray image', 400, 400)
    cv2.waitKey(1)

    screen = screen[y:y+h, x:x+w]
    screen = torch.from_numpy(screen)

    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).to(device)


BATCH_SIZE = 50
GAMMA = 0.999
EPS_START = 1
EPS_END = 0.1
EPS_DECAY = 2000
TARGET_UPDATE = 10

init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape

# Get number of actions from gym action space
n_actions = env.action_space.n

policy_net = Googles_DQN.DQN(screen_height, screen_width, n_actions).to(device)
policy_net.apply(Googles_DQN.DQN.weights_init_uniform_rule)
target_net = Googles_DQN.DQN(screen_height, screen_width, n_actions).to(device)
print(target_net)
# GOOGLE: Initiliaze replay memory D to capacity N
memory = ReplayMemory(50000)

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

optimizer = optim.RMSprop(policy_net.parameters())


steps_done = 0

def select_action(state):
    global steps_done
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


def plot_durations(save_fig=False):
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())

    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if save_fig:
        plt.savefig("./durations_complete"+str(seed) + ".png")
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


def plot_rewards(save_fig=False):
    plt.figure(3)
    plt.clf()
    rewards = torch.tensor(cumulative_reward, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(rewards.numpy())

    # Take 100 episode averages and plot them too
    if len(rewards) >= 100:
        means = rewards.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if save_fig:
        plt.savefig("./rewards_complete"+str(seed) + ".png")
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


def plot_loss(save_fig=False):
    plt.figure(3)
    plt.clf()
    loss = torch.tensor(running_loss, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.plot(loss.numpy())

    # Take 100 episode averages and plot them too
    if len(loss) >= 100:
        means = loss.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if save_fig:
        plt.savefig("./loss_complete"+str(seed) + ".png")
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


def optimize_model():
    if len(memory) < BATCH_SIZE or len(memory) < start_learning:
        return
    # GOOGLE: Line 11
    for s in range(BATCH_SIZE):
        index = random.randint(0, len(memory) - 5)
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
                t = memory.sample(index+idx+1)
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

                reward_batch[0] = 1
                for i, rw in enumerate(reward_batch[:-1]):
                    reward_batch[i+1] = reward_batch[i]*(gamma**(i+1))
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
num_episodes = 500
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
        print(reward, done, info['ale.lives'])
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
            print("done")
        # GOOGLE: line 10
        # Store the transition in memory
        memory.push(state, action, next_state, reward, done)

        # Move to the next state
        state = next_state
        temp_reward += reward.data[0]
        # Perform one step of the optimization (on the target network)
        env.render()
        if train:
            optimize_model()

        if done:
            cumulative_reward.append(temp_reward)
            episode_durations.append(t + 1)
            plot_durations(save_fig=True)
            plot_rewards(save_fig=True)
            plot_loss()
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
plot_durations(save_fig=True)
plot_rewards(save_fig=True)

torch.save(policy_net.state_dict(), './model/modelcomplete.pyt')
with open('./model/modelMemory'+str(seed) + '.pkl', 'wb') as output:
    pickle.dump(memory, output, pickle.HIGHEST_PROTOCOL)
with open('./model/cumulative_rewards'+str(seed) + '.pkl', 'wb') as output:
    pickle.dump(cumulative_reward, output, pickle.HIGHEST_PROTOCOL)
with open('./model/episode_durations'+str(seed) + '.pkl', 'wb') as output:
    pickle.dump(episode_durations, output, pickle.HIGHEST_PROTOCOL)
env.close()
plt.ioff()


