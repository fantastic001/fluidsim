from .BaseAnimator import BaseAnimator

class DebugAnimator(BaseAnimator):
    
    def start(self, simulator, **kwargs):
        pass

    def update(self, p, v, b, iteration):
        print("Iteration " + str(iteration))
        f = open("iter-" + str(iteration) + ".txt", "w")
        n,m = p.shape
        for i in range(n):
            for j in range(m):
                f.write("f[" + str(i) + "]["+str(j)+"] = ")
                for a in range(9):
                    f.write(str(self.simulator.f[i][j][a]) + " ")
                f.write("\n")
        f.close()

