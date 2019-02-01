# Rainbow_RL
Reinforcement Learning Project implementing DQN and variations on a simple environment, training on basic CPU, based on
[Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298)

## Environment

A mouse (white square in the examples below) moves on a grid. It only sees the
5x5 grid around itself and has to eat as much cheese (+0.5 reward, red squares in examples) as possible,
avoiding the poisons (-1 reward, blue squares in examples).
We also introduced moving cats (-100 reward, green squares in examples), that is needs to avoid.


## Command lines
To train from scratch or test :
```shell
python3 main.py --s [solver] -a [action] -n [number of hunters]
```

* [solver] can be : "dqn", "double_dqn", "dueling_dqn", "prioritized_dqn", "distributional_dqn", "noisy_dqn", "multistep_dqn" or "human"

* [action] can be : "test", to test an already existing model or "validate", to train and test a model from scratch

* [number of hunters] is be any non-negative integer


At every test and train, videos of games (.mp4), models weights (.json and .h5) and performance vector (.pkl) will automatically be saved in the repository

To plot the analysis curves :
```shell
python3 plot.py
```

## Examples

### Example of DoubleDQN agent in a 25x25 grid

![](gif/double_dqn9.gif)

### Example of DQN agent in a 10-cats (hunters) environment

![](gif/dqn180.gif)

### Example of PrioritizedDQN agent in a 13x13 grid

![](gif/prioritized_dqn190.gif)
