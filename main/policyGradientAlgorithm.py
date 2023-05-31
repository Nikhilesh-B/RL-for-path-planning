import tensorflow as tf
from keras.layers import Dense, InputLayer
from tensorflow import keras
from constants import OBSTACLES, START, GOAL, LINK_LENGTH
from tensorflow_probability import distributions

tfd = distributions


class policyGradientAlgorithm():
    # Initializing the class and its variables.
    def __init__(self, learning_rate=0.001, gamma=0.003, horizon=4, inputLayer_dims=2, fc1_dims=1, fc2_dims=10,
                 output_dims=2):
        # Define instance variables related to rewards, actions, state history, and the network.
        self.discountedRewardHistory = None
        self.reward_history = None
        self.action_history = None
        self.state_history = None
        self.network = None
        self.output = None
        self.fc2 = None
        self.fc1 = None
        self.inputLayer = None
        self.gamma = gamma
        self.horizon = horizon
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.inputLayer_dims = inputLayer_dims
        self.output_dims = output_dims
        self.learning_rate = learning_rate

    # Constructs the neural network using Keras' Sequential API
    def constructNN(self):
        # Define the structure of the network with input layer, two dense layers and an output layer
        self.inputLayer = InputLayer(input_shape=(self.inputLayer_dims,))
        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.output = Dense(self.output_dims, activation='relu')

        self.network = keras.models.Sequential([
            self.inputLayer,
            self.fc1,
            self.output
        ])

        # Initialize history variables for states, actions and rewards
        self.state_history = [START]
        self.action_history = [START]
        self.reward_history = [0]
        self.discountedRewardHistory = None

    # Gets the last action taken
    def getPrevAction(self):
        return [self.action_history[-1]]

    # Computes the next action using multivariate normal distribution.
    @staticmethod
    def getNextAction(newState):
        mu = newState[0]
        sigma = [[1, 0], [0, 1]]
        actionDist = tfd.MultivariateNormalFullCovariance(loc=mu, covariance_matrix=sigma)
        nextAction, logPdf = actionDist.experimental_sample_and_log_prob()
        return nextAction, logPdf

    # Checks if a robot arm configuration would result in a collision with an obstacle.
    def detect_collision(self, arm, config, obstacles):
        arm.update_joints(config)
        points = arm.points
        for k in range(len(points) - 1):
            for circle in obstacles:
                a_vec = tf.constant(points[k])
                b_vec = tf.constant(points[k + 1])
                c_vec = tf.constant([circle[0], circle[1]])
                radius = circle[2]

                line_vec = b_vec - a_vec
                line_mag = tf.norm(line_vec)
                circle_vec = c_vec - a_vec
                proj = (1 / line_mag) * tf.tensordot(circle_vec, line_vec)

                if proj <= 0:
                    closest_point = a_vec
                elif proj >= line_mag:
                    closest_point = b_vec
                else:
                    closest_point = a_vec + line_vec * proj / line_mag

                if tf.norm(closest_point - c_vec) <= radius:
                    return True

        return False

    # Checks if the end effector of the arm is close to the goal.
    @staticmethod
    def closeToGoal(newAction, Arm, threshold):
        theta0 = newAction[0][0]
        theta1 = newAction[0][1]

        link1_length, link2_length = Arm.link_lengths[0], Arm.link_lengths[1]

        x_joint1, y_joint1 = link1_length * tf.math.cos(theta0), link1_length * tf.math.sin(theta0)
        x_joint2, y_joint2 = x_joint1 + link2_length * tf.math.cos(theta1), y_joint1 + link2_length * tf.math.sin(
            theta1)

        endEffectorPosition = tf.concat([x_joint2, y_joint2], axis=0)

        g = tf.constant(GOAL)
        goalFunction = tfd.MultivariateNormalFullCovariance(loc=endEffectorPosition, covariance_matrix=[[1, 0], [0, 1]])
        goalPotentialFunctionValue = 1000 * goalFunction.prob(newAction)

        isCloseToGoal = (tf.norm(endEffectorPosition - g) <= threshold)

        return goalPotentialFunctionValue, isCloseToGoal

    # Computes the reward for a given action.
    def getNextReward(self, newAction, Arm):
        reward = 0

        if self.detect_collision(arm=Arm, config=newAction, obstacles=OBSTACLES):
            reward -= 1000

        goalPotentialFunction, isCloseToGoal = self.closeToGoal(newAction=newAction, Arm=Arm, threshold=0.3)
        reward += goalPotentialFunction

        reward = tf.constant(reward)
        return reward

    # Resets the experiment and clears the history of states, actions, and rewards.
    def resetExperiment(self):
        self.state_history = [START]
        self.action_history = [START]
        self.reward_history = [0]

    # Stores a new transition in the history.
    def storeTransition(self, newState, newAction, newReward):
        self.state_history.append(newState)
        self.action_history.append(newAction)
        self.reward_history.append(newReward)

    # Getter methods for history of states, actions and discounted rewards.
    def getStateHistory(self):
        return self.state_history

    def getActionHistory(self):
        return self.action_history

    def getDiscountedRewardHistory(self):
        return self.discountedRewardHistory

    # Prints the summary of the neural network.
    def print(self):
        print(" XXXXXXXXXXXXXXX ")
        self.network.summary()
        print(" XXXXXXXXXXXXXXX ")

    # Calculates the discounted reward for each state in the history.
    def computeDiscountedReward(self):
        self.discountedRewardHistory = [0] * len(self.state_history)
        for k in range(0, len(self.reward_history)):
            val = 0
            for t in range(k, len(self.reward_history)):
                val += self.reward_history[t] * (self.gamma ** (t - k))

            self.discountedRewardHistory[k] = val

        self.discountedRewardHistory = tf.convert_to_tensor(self.discountedRewardHistory)

        return self.discountedRewardHistory

    # Computes the policy for a given reward and log probability of an action.
    @staticmethod
    def computePolicy(reward, logPdf):
        return reward * logPdf
