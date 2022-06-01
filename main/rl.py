from nlinkarm import NLinkArm
from helper import visualize_spaces, animate, find_qnew, find_qnew_greedy, clear_path
import numpy as np
from pprint import pprint 


OBSTACLES = [[1.75, 0.75, 0.6], 
             [-.5, 1.5, 0.5], 
             [0, -1, 0.7]] 
START = (1.0, 0.0)    
GOAL = (1.0, 0.5)    

LINK_LENGTH = [1, 1] 
MAX_NODES = 400 
BIAS = 0.05
DELTA = 0.2      
EDGE_INC = 0.05  



def main():
    ARM = NLinkArm(LINK_LENGTH, [0,0])
    visualize_spaces(ARM, START, OBSTACLES)

    roadmap, route = construct_tree(ARM)

    print("Roadmap")
    pprint(roadmap)
    print("Route")
    pprint(route)

    if not route:
        print("No route found")

    animate(ARM, roadmap, route, START, OBSTACLES) 



def construct_tree(arm):
    """
    :param arm: NLinkArm object
    :return: roadmap: Dictionary of nodes in the constructed tree {(node_x, node_y): (parent_x, parent_y)}
    :return: path: List of configurations traversed from start to goal
    """
    tree = {START: None}
    path = []

    count = 0
    while (len(tree) < MAX_NODES and (GOAL not in tree)):
      qrand_theta1 = np.random.uniform(-1*np.pi, np.pi)
      qrand_theta2 = np.random.uniform(-1*np.pi, np.pi)

      qrand = (qrand_theta1, qrand_theta2)

      biasDraw = np.random.binomial(1, BIAS, 1)
      if (biasDraw==1):
        qrand = GOAL
      
      #qnear, qnew = find_qnew(tree, qrand)
      qnear, qnew = find_qnew(arm, tree, qrand, DELTA)

      if (clear_path(arm, qnew, qnear, EDGE_INC)):
        tree[qnew] = qnear
        
    if (GOAL in tree):
        path = [GOAL]
        pathNode = GOAL

        while(pathNode != START):
          parent = tree[pathNode]
          path.append(parent)
          pathNode = parent
        
        path = path[::-1]
    
    return tree, path



if __name__ == "__main__":
    main()