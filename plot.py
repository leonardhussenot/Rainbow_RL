import pickle
import numpy as np
import matplotlib.pyplot as plt

#========= Graph 1 : Comparaion des différents modeles sur 200 itérations 

racine = './models_graph1/'

with open(racine + "dqn",'rb') as file:
    dqn = pickle.load(file)
    print(dqn)

with open(racine + "multistep_dqn_n=2",'rb') as file:
    multistep_dqn = pickle.load(file)

with open(racine + "double_dqn",'rb') as file:
    double_dqn = pickle.load(file)

with open(racine + "prioritized_dqn",'rb') as file:
    prioritized_dqn = pickle.load(file)
    
with open(racine + "human",'rb') as file:
    human = pickle.load(file)
    
with open(racine + "noisy_dqn_1",'rb') as file:
    noisy = pickle.load(file)
    
with open(racine + "dueling_dqn",'rb') as file:
    dueling_dqn = pickle.load(file)

#legends=['dqn','prioritized_dqn','double_dqn','multistep_dqn','human','noisy','dueling_dqn']
legends = ['w = 0', 'w = 0.5']

plt.plot(dqn,linewidth=2.0)
plt.plot(prioritized_dqn,linewidth=2.0)
#plt.plot(double_dqn,linewidth=2.0)
#plt.plot(multistep_dqn,linewidth=2.0)
#plt.plot(human,linewidth=2.0)
#plt.plot(noisy,linewidth=2.0)
#plt.plot(dueling_dqn,linewidth=2.0)

plt.legend(legends)

plt.xlabel('training iterations (x10)')
plt.ylabel('test performances')

plt.show()

#========= Graph 3 : Multistep n=2,3,5

racine = './models_graph3/'

with open(racine + "multistep_dqn_n=2",'rb') as file:
    multistep_dqn2 = pickle.load(file)

with open(racine + "multistep_dqn_n=3",'rb') as file:
    multistep_dqn3 = pickle.load(file)

with open(racine + "multistep_dqn_n=5",'rb') as file:
    multistep_dqn5 = pickle.load(file)
    
legends=['n = 2','n = 3','n = 4']

plt.plot(multistep_dqn2,linewidth=2.0)
plt.plot(multistep_dqn3,linewidth=2.0)
plt.plot(multistep_dqn5,linewidth=2.0)

plt.legend(legends)

plt.xlabel('training iterations (x10)')
plt.ylabel('test performances')
plt.title('DQN Multistep')

plt.show()

#========= Graph 4 : NoisyNets std=0,1,2

racine = './models_graph4/'

with open(racine + "noisy_dqn_1",'rb') as file:
    noisy1 = pickle.load(file)

with open(racine + "noisy_dqn_05",'rb') as file:
    noisy0 = pickle.load(file)

with open(racine + "noisy_dqn_2",'rb') as file:
    noisy2 = pickle.load(file)
    
legends=['std = 0.5','std = 1','std = 2']

plt.plot(noisy0,linewidth=2.0)
plt.plot(noisy1,linewidth=2.0)
plt.plot(noisy2,linewidth=2.0)

plt.legend(legends)

plt.xlabel('training iterations (x10)')
plt.ylabel('test performances')
plt.title('noisy nets')

plt.show()

#=============== Graph 5: Test on a 25x25 grid

dqn = [40,70.5,63,31.5,43,39.5,66,60,48,61.5]
double_dqn = [58,70,20,60,48.5,68,46,61.5,63,81.5]
human = [24,39,26.5,18.5,15.5,15.5,36,32,24,22]
prioritized_dqn = [47, 60.5, 46, 39.5, 24, 34, 59.5, 32, 57.5, 30.5]
noisy_net = [44, 59.5, 46.5, 29.5, 64, 57.5, 72.5, 38, 37.5, 62]
multistep_n2 = [67.5, 50, 50, 59, 73, 61, 72, 66, 40, 72.5]  

BoxName = ['','dqn','double_dqn','human','prioritized', 'noisy', 'multistep','']

data = [dqn, double_dqn, human, prioritized_dqn, noisy_net, multistep_n2]

plt.boxplot(data)
plt.xticks(np.arange(len(BoxName)-1),BoxName,rotation=20)
plt.title('Final performances on 25 x 25 grid')
plt.show()
