
from .BaseLogger import BaseLogger

import matplotlib.pyplot as plt


class StandardOutputLogger(BaseLogger):

    def start(self, **kwargs):
        self.SAVE_STEPS = kwargs.get("save_steps", False)
        self.SAVE_DIR = kwargs.get("save_dir", "steps")
        self.BOUND_VALUE = kwargs.get("bound_value", 20)

    def finish(self):
        pass

    def my_print(self, string):
        print(string)

    def my_break(self):
        input(">> Hit ENTER to cintinue ")

    def my_plot_field(self, v, title):
        if len(v.shape) > 2:
            v_ = v / self.BOUND_VALUE
            plt.quiver(v_[::10,::10, 0], v_[::10,::10,1], units="width")
            plt.title(title)
            if self.SAVE_STEPS:
                plt.savefig(self.SAVE_DIR + "/" + title + ".png")
            else:
                plt.show()
            plt.clf()
        else:
            plt.title(title)
            plt.imshow(v)
            plt.show()
