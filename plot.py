import pickle
import numpy as np
import matplotlib.pyplot as plt


with open("./models/dqn",'rb') as file:
    dqn = pickle.load(file)
    print(dqn)

with open("./models/dueling_dqn",'rb') as file:
    dueling_dqn = pickle.load(file)
    print(dueling_dqn)

with open("./models/double_dqn",'rb') as file:
    double_dqn = pickle.load(file)

with open("./models/prioritized_dqn",'rb') as file:
    prioritized_dqn = pickle.load(file)

legends=['dqn','prioritized_dqn','double_dqn','dueling_dqn']

plt.plot(dqn)
plt.plot(prioritized_dqn)
plt.plot(double_dqn)
plt.plot(dueling_dqn)

plt.legend(legends)

plt.xlabel('training iterations')
plt.ylabel('test performances')

plt.show()