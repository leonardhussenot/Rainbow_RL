import numpy as np

class Memory(object):
    def __init__(self, max_memory=100):
        self.max_memory = max_memory
        self.memory = list()

    def remember(self, m):
        if len(self.memory) < self.max_memory:
            self.memory.append(m)
        else:
            self.memory = self.memory[1:]
            self.memory.append(m)


    def random_access(self):
        r = np.random.randint(0,len(self.memory))
        return self.memory[r]

class Memory_prioritized(Memory):
    def __init__(self, max_memory=100):
        super(Memory_prioritized, self).__init__(max_memory=100)
        self.priority = list()
        
    def prioritised_remember(self, m, model, discount):
        """Remember both (state, next_state, action, reward, game_over) tuple and a score associated to this training example"""
        s = m[0]
        n_s = m[1]
        a = m[2]
        r = m[3]
        game_over = m[4]
        
        target = r if game_over else r + discount * np.amax(model.predict(np.array([n_s])))
        output = model.predict(np.array([s]))[0,a]
        
        score = np.abs(target - output)**0.5 # L'exposant est un hyper paramètre
        
        
        if len(self.memory)<self.max_memory:
            self.memory.append(m)
            self.priority.append(0.1+score) # La constante 0.1 est un hyper paramètre
        else:
            self.memory=self.memory[1:]
            self.priority = self.priority[1:]
            self.memory.append(m)
            self.priority.append(score)
    
    def prioritized_access(self):
        """ Choose a training example with a probability proportional to its score and not randomly """
        indice = np.argmax(np.random.multinomial(1, (self.priority)/np.sum(self.priority), size=1))
        return self.memory[indice]