
from .BaseLogger import BaseLogger



class StandardOutputLogger(BaseLogger):

    def start(self, **kwargs):
        pass

    def finish(self):
        pass

    def my_print(self, string):
        print(string)

    def my_break(self):
        input(">> Hit ENTER to cintinue ")

    def my_plot_field(self, v, title):
        if len(v.shape) > 2:
            plt.quiver(v[::10,::10, 0], v[::10,::10,1], units="width")
            plt.title(title)
            plt.show()
            plt.clf()
        else:
            plt.title(title)
            plt.imshow(v)
            plt.show()
