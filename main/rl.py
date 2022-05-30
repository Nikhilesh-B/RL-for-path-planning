from helper import NLinkArm, visualize_spaces, animate






OBSTACLES = [[1.75, 0.75, 0.6], [-.55, 1.5, 0.5], [0, -1, 0.7], [-2, -0.5, 0.6]]
START = (-3.0, 1.0)
GOAL = (-0.5, 0.5)
LINK_LENGTH = [1, 1] 




def main():
    ARM = NLinkArm(LINK_LENGTH, [0,0])
    visualize_spaces(ARM)

    #animate(ARM, roadmap, route, START, ) only have to do the animation after learning the network
