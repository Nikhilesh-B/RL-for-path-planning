import tensorflow as tf



class robotEnv():
    def __init__(self, obstacles=None, start=None, goal=None, link_length=None):
        self.obstacle = obstacles
        self.start = start
        self.goal = goal
        self.link_length = link_length


