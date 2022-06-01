import numpy as np
import matplotlib.pyplot as plt

from rl import OBSTACLES


"""
Class for an n-link arm
"""
class NLinkArm(object):
    def __init__(self, link_lengths, joint_angles):
        self.n_links = len(link_lengths)
        if self.n_links != len(joint_angles):
            raise ValueError()

        self.link_lengths = np.array(link_lengths)
        self.joint_angles = np.array(joint_angles)
        self.points = [[0, 0] for _ in range(self.n_links + 1)]

        self.lim = sum(link_lengths)
        self.update_points()

    def update_joints(self, joint_angles):
        self.joint_angles = np.array(joint_angles)
        self.update_points()

    def update_points(self):
        for i in range(1, self.n_links + 1):
            self.points[i][0] = self.points[i - 1][0] + \
                self.link_lengths[i - 1] * \
                np.cos(np.sum(self.joint_angles[:i]))
            self.points[i][1] = self.points[i - 1][1] + \
                self.link_lengths[i - 1] * \
                np.sin(np.sum(self.joint_angles[:i]))

        self.end_effector = np.array(self.points[self.n_links]).T


"""
Utility functions
"""

def detect_collision(arm, config, OBSTACLES):
    """
    :param arm: NLinkArm object
    :param config: Configuration (joint angles) of the arm
    :return: True if any part of arm collides with obstacles, False otherwise
    """
    arm.update_joints(config)
    points = arm.points
    for k in range(len(points) - 1):
        for circle in OBSTACLES:
            a_vec = np.array(points[k])
            b_vec = np.array(points[k+1])
            c_vec = np.array([circle[0], circle[1]])
            radius = circle[2]

            line_vec = b_vec - a_vec
            line_mag = np.linalg.norm(line_vec)
            circle_vec = c_vec - a_vec
            proj = circle_vec.dot(line_vec / line_mag)

            if proj <= 0:
                closest_point = a_vec
            elif proj >= line_mag:
                closest_point = b_vec
            else:
                closest_point = a_vec + line_vec * proj / line_mag

            if np.linalg.norm(closest_point - c_vec) <= radius:
                return True

    return False


def closest_euclidean(q, qp):
    """
    :param q, qp: Two 2D vectors in S1 x S1
    :return: qpp, dist. qpp is transformed version of qp so that L1 Euclidean distance between q and qpp
    is equal to toroidal distance between q and qp. dist is the corresponding distance.
    """
    q = np.array(q)
    qp = np.array(qp)

    A = np.meshgrid([-1,0,1], [-1,0,1])
    qpp_set = qp + 2*np.pi*np.array(A).T.reshape(-1,2)
    distances = np.linalg.norm(qpp_set-q, 1, axis=1)
    ind = np.argmin(distances)
    dist = np.min(distances)

    return qpp_set[ind], dist













"""
Plotting and visualization functions
"""

def get_occupancy_grid(arm, M, OBSTACLES):
    grid = [[0 for _ in range(M)] for _ in range(M)]
    theta_list = [2 * i * np.pi / M for i in range(-M // 2, M // 2 + 1)]
    for i in range(M):
        for j in range(M):
            grid[i][j] = int(detect_collision(arm, [theta_list[i], theta_list[j]], OBSTACLES))
    return np.array(grid)

def plot_roadmap(ax, roadmap):
    for node, parent in roadmap.items():
        if parent is not None:
            euc_parent, _ = closest_euclidean(node, parent)
            euc_node, _ = closest_euclidean(parent, node)
            ax.plot([node[0], euc_parent[0]], [node[1], euc_parent[1]], "-k")
            ax.plot([euc_node[0], parent[0]], [euc_node[1], parent[1]], "-k")
        ax.plot(node[0], node[1], ".b")

def plot_arm(plt, ax, arm, OBSTACLES):
    for obstacle in OBSTACLES:
        circle = plt.Circle((obstacle[0], obstacle[1]), radius=obstacle[2], fc='k')
        plt.gca().add_patch(circle)

    for i in range(arm.n_links + 1):
        if i is not arm.n_links:
            ax.plot([arm.points[i][0], arm.points[i + 1][0]], [arm.points[i][1], arm.points[i + 1][1]], 'r-')
        ax.plot(arm.points[i][0], arm.points[i][1], 'k.')

def visualize_spaces(arm, START, OBSTACLES):
    plt.subplots(1, 2)

    plt.subplot(1, 2, 1)
    grid = get_occupancy_grid(arm, 200, OBSTACLES)
    plt.imshow(np.flip(grid.T, axis=0))
    plt.xticks([0,50,100,150,200], ["-\u03C0", "-\u03C0/2", "0", "\u03C0/2", "\u03C0"])
    plt.yticks([0,50,100,150,200], ["\u03C0", "\u03C0/2", "0", "-\u03C0/2", "-\u03C0"])
    plt.title("Configuration space")
    plt.xlabel('joint 1')
    plt.ylabel('joint 2')

    ax = plt.subplot(1, 2, 2)
    arm.update_joints(START)
    plot_arm(plt, ax, arm)
    plt.title("Workspace")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('scaled')
    plt.xlim(-3.0, 3.0)
    plt.ylim(-3.0, 3.0)
    plt.show()

def clear_path(arm, q1, q2, EDGE_INC):
    """
    :param arm: NLinkArm object
    :param q1, q2: Two configurations in S1 x S1
    :return: True if edge between q1 and q2 sampled at EDGE_INC increments collides with obstacles, False otherwise
    """

    #You need to call update joints here, in the direction fo
    q2, dist = closest_euclidean(q1,q2)
    #above we have the shortest euclidean to q2 above that we can then play with 
    q1_theta1 = q1[0]
    q1_theta2 = q1[1]

    q2_theta1 = q2[0]
    q2_theta2 = q2[1]

    if ((q2_theta1-q1_theta1) == 0):
        alpha = (np.pi)/2
    else:
      m = (q2_theta2-q1_theta2)/(q2_theta1-q1_theta1)
      alpha = np.arctan(m)

    

    #How do you know which gradient to apply the above too?
    

    if (q2_theta1 > q1_theta1):
        while (q1_theta1<q2_theta1):
          q1_theta1 = q1_theta1+EDGE_INC*np.cos(alpha)
          q1_theta2 = q1_theta2+EDGE_INC*np.sin(alpha)
          if(detect_collision(arm, (q1_theta1, q1_theta2))):
            return False

    else:
        while (q1_theta1>q2_theta1):
          q1_theta1 = q1_theta1-EDGE_INC*np.cos(alpha)
          q1_theta2 = q1_theta2+EDGE_INC*np.sin(alpha)
          if(detect_collision(arm, (q1_theta1, q1_theta2))):
            return False    

    return True



def find_qnew(tree, qrand, DELTA):
    """
    :param tree: RRT dictionary {(node_x, node_y): (parent_x, parent_y)}
    :param qrand: Randomly sampled configuration
    :return: qnear in tree, qnew between qnear and qrand with distance DELTA from qnear
    """
    min_dist = np.inf 
    qnear = None 

    #finding the nearest node
    for key in tree.keys():
      qrandpp, dist = closest_euclidean(qrand, key)

      if (dist < min_dist):
          min_dist = dist
          qnear = key
    
    #adding the node on the way to the nearest node 
    #what do you do if you have a collision here


    #what if you cannot connect the node here?
    if (min_dist <= DELTA):
      #if (not qrandpp.any(None)):
      return qnear, qrandpp

      #else: 
      #  return qnear, qrand


    #what if you cannot connect the node here? (maybe we can check for this on the macro level)
    else:
      q1_theta1 = qnear[0] 
      q1_theta2 = qnear[1]

      #if (not qrandpp.any(None)):
      q2_theta1 = qrandpp[0] 
      q2_theta2 = qrandpp[1]
      
      #else: 
      #  q2_theta1 = qrand[0]
      #  q2_theta2 = qrand[1]


      if ((q2_theta1-q1_theta1) == 0):
        alpha = (np.pi)/2
      
      else:
        m = (q2_theta2-q1_theta2)/(q2_theta1-q1_theta1)
        alpha = np.arctan(m)

      if (q1_theta1<q2_theta1):
        qnew_theta1 = q1_theta1+DELTA*np.cos(alpha)
      else: 
        qnew_theta1 = q1_theta1-DELTA*np.cos(alpha)
      
      qnew_theta2 = q1_theta2+DELTA*np.sin(alpha)

      qnew = (qnew_theta1, qnew_theta2)
      return qnear, qnew


def find_qnew_greedy(arm, tree, qrand, DELTA):
    """
    :param arm: NLinkArm object
    :param tree: RRT dictionary {(node_x, node_y): (parent_x, parent_y)}
    :param qrand: Randomly sampled configuration
    :return: qnear in tree, qnew between qnear and qrand as close as possible to qrand in increments of DELTA
    """
    qnear, qnew = find_qnew(tree, qrand)

    q1_theta1 = qnew[0]
    q1_theta2 = qnew[1]

    q2_theta1 = qrand[0]
    q2_theta2 = qrand[1]

    if ((q2_theta1-q1_theta1) == 0):
        alpha = (np.pi)/2
    else:
      m = (q2_theta2-q1_theta2)/(q2_theta1-q1_theta1)
      alpha = np.arctan(m)



    if (q2_theta1 > q1_theta1):
        while (q1_theta1<q2_theta1):
          q1_theta1 = q1_theta1+DELTA*np.cos(alpha)
          q1_theta2 = q1_theta2+DELTA*np.sin(alpha)
          qnew = (q1_theta1, q1_theta2)
          
          if(clear_path(arm, qnear, qnew)):
            continue

          else:
            return qnear, qnew
       
        return qnear, qrand  

    else:
        while (q1_theta1>q2_theta1):
          q1_theta1 = q1_theta1-DELTA*np.cos(alpha)
          q1_theta2 = q1_theta2+DELTA*np.sin(alpha)
          qnew = (q1_theta1, q1_theta2)

          if(clear_path(arm, qnear, qnew)):
            continue

          else:
            return qnear, qnew
        
        return qnear, qrand  





def animate(arm, roadmap, route, START, OBSTACLES):
    ax1 = plt.subplot(1, 2, 1)
    plot_roadmap(ax1, roadmap)
    if route:
        plt.plot(route[0][0], route[0][1], "Xc")
        plt.plot(route[-1][0], route[-1][1], "Xc")
    plt.title("Configuration space")
    plt.xlabel('joint 1')
    plt.ylabel('joint 2')
    plt.axis('scaled')
    plt.xlim(-3.2, 3.2)
    plt.ylim(-3.2, 3.2)

    ax2 = plt.subplot(1, 2, 2)
    arm.update_joints(START)
    plot_arm(plt, ax2, arm)
    plt.title("Workspace")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('scaled')
    plt.xlim(-3.0, 3.0)
    plt.ylim(-3.0, 3.0)
    plt.pause(1)


    print("here's the config",route)
    for config in route:
        print(config)
        arm.update_joints([config[0], config[1]])
        ax1.plot(config[0], config[1], "xr")
        ax2.lines = []
        plot_arm(plt, ax2, arm, OBSTACLES)
        # Uncomment here to save the sequence of frames
        # plt.savefig('frame{:04d}.png'.format(i))
        plt.pause(0.3)

    plt.show()