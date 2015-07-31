
class BaseAnimator(object):
    
    def __init__(self, simulator, **kwargs):
        self.simulator = simulator 
        self.start(self.simulator, **kwargs)
    
    def start(self, simulator, **kwargs):
        pass

    def update(self, p, v, b, iteration):
        pass 
    
    def finish(self):
        pass

    def run(self, iters, step=0.1):
        p,v,b = self.simulator.data()
        self.update(p, v, b, 0)
        for i in range(iters):
            self.simulator.step(step)
            p,v,b = self.simulator.data()
            self.update(p, v, b, i+1)
        self.simulator.finish()
        self.finish()
