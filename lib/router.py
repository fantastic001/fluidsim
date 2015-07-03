
class Router(object):
    
    def __init__(self):
        self.choices = [] 

    def register(self, choice, name):
        self.choices.append((choice, name))

    def route(self, name):
        for poschoice in self.choices:
            posobj, posname = poschoice
            if name == posname:
                return posobj
