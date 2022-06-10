import tensorflow as tf



class robotEnv():
    def __init__(self, obstacles=None, start=None, goal=None, link_length=None):
        self.obstacles = obstacles
        self.start = start
        self.goal = goal
        self.link_length = link_length

    def __print__(self):
        print("obstacles=",self.obstacles)
        print("start=",self.start)
        print("goal=",self.goal)
        print("link_length=",self.link_length)
