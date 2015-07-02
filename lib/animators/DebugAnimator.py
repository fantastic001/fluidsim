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
                    if i<10 and i>=5:
                        # THIS IS OUT 
                        if self.simulator.f[i][j][a] != self.simulator.f[i-5][j][a]:
                            print("HERE WENT WRONG ON " + str(i) + " " + str(j))
                f.write("\n")
        f.close()

