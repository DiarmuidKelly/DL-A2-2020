import torch
import matplotlib
import matplotlib.pyplot as plt
from IPython import display


def plot_durations(is_ipython, episode_durations, seed, save_fig=False):
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


def plot_rewards(is_ipython, cumulative_reward, seed, save_fig=False):
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


def plot_loss(is_ipython, running_loss, seed, save_fig=False):
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
