# Running Atari DQN in OpenAI Gym using Pytorch
##### Variable Parameters:
Batch Size [-b] - 32 (maybe more) 
```
Note that when sampling from memory 32 samples, each sample is of batch size (defualt=4)
So, doubling this number increases the number of samples by the increase times 4, computational cost
```
Number of episodes [-ep] - 2000 default
```
Note each episode takes roughly 30 seconds and is on average 160 frames in the beginning
10,000 steps will take roughly 83 hours
```
Learning Rate [-lr] - 0.00025 Default

Negative Rewards [-neg] - Default False

# Usage

```bash
usage: main.py [-h] [--peregrine] [--load] [--batch_size BATCH_SIZE]
               [--learning_rate LEARNING_RATE] [--episodes EPISODES]
               [--negative_rewards]

Run DQN Atari

optional arguments:
  -h, --help            show this help message and exit
  --peregrine, -p       Set condition for Peregrine Environment (Default:
                        False)
  --load, -l            Load the model from memory (Default: False)
  --batch_size BATCH_SIZE, -b BATCH_SIZE
                        Batch Size (Default: 4)
  --learning_rate LEARNING_RATE, -lr LEARNING_RATE
                        Learning Rate (Default: 0.00025)
  --episodes EPISODES, -ep EPISODES
                        Number of Episodes (Default: 2000)
  --negative_rewards, -neg
                        Use negatives rewards in training (Default: False)
```

