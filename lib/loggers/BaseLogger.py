
class BaseLogger(object):
    

    def __init__(self, **kwargs):
        self.DEBUG = kwargs.get("debug", False)
        self.DEBUG_BREAK = kwargs.get("debug_break", False)
        self.DEBUG_INTERACTIVE = kwargs.get("debug_interactive")
        self.start(**kwargs)

    def start(self, **kwargs):
        pass

    def finish(self):
        pass

    def my_print(self, string):
        pass

    def my_break(self):
        pass

    def my_plot_field(self, v, title):
        pass

    def plot_field(self, v, title=""):
        if self.DEBUG_INTERACTIVE:
            self.my_plot_field(v, title)
        if self.DEBUG_BREAK:
            self.my_break()

    def print_vector(self, s, v, full=False):
        if self.DEBUG:
            self.my_print(s)
            if full:
                self.my_print(str(v.tolist()))
            else:
                self.my_print(str(v))
            if self.DEBUG_BREAK:
                self.my_break()
