from .BaseAnimator import BaseAnimator

class DebugAnimator(BaseAnimator):
    
    def start(self, simulator):
        pass

    def update(self, grid, iteration):
        print("Iteration " + str(iteration))
        f = open("iter-" + str(iteration) + ".txt", "w")
        n,m = len(grid), len(grid[0])
        for i in range(n):
            for j in range(m):
                f.write("f[" + str(i) + "]["+str(j)+"] = ")
                for a in range(9):
                    f.write(str(self.simulator.f[i][j][a]) + " ")
                f.write("\n")
        f.close()

