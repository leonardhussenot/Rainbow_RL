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
        r=np.random.randint(0,len(self.memory))
        return self.memory[r]
        
