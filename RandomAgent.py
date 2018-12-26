import Agent

class RandomAgent(Agent):
    def __init__(self):
        super(RandomAgent, self).__init__()
        pass

    def learned_act(self, s):
        rand = np.random.randint(0,4)
        return rand
