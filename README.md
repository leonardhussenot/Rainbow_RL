# Rainbow_RL
Project on RAINBOW paper, reinforcement learning

To train from scratch or test : 

python3 main.py --s [solver] –-a [action] –-n [number of hunters]


[solver] can be : “dqn”, “double_dqn”, “duelling_dqn”, “prioritized_dqn”, “distributional_dqn”, “noisy_dqn”, “human”, “multistep_dqn”

[action] can be : “test”, to test an already existing model or “validate”, to train and test a model from scratch

[number of hunters] can be any non negative integer



At every test and train, videos of games (mp4), models weights (json and h5) and performance vector (pkl) will automatically be saved in the repository
 

### DQN with cats (hunters)

![](gif/dqn180.gif)

### DoubleDQN

![](gif/double_dqn9.gif)

### PrioritizedDQN

![](gif/prioritized_dqn190.gif)
