from nlinkarm import NLinkArm
from helper import visualize_spaces, animate, detect_collision
from constants import OBSTACLES, START, GOAL, LINK_LENGTH
from tensorflow_probability import distributions as tfd
from keras.layers import Dense, InputLayer
from tensorflow import keras
from robotEnv import robotEnv
from pprint import pprint
import math as m
import tensorflow as tf


class policyGradientAlgorithm():
    def __init__(self, robot_environment, robot_arm, learning_rate=0.001, horizon=4, inputLayer_dims=2, fc1_dims=1,fc2_dims=1, output_dims=2):
        self.horizon = horizon
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.inputLayer_dims = inputLayer_dims
        self.output_dims = output_dims
        self.learning_rate = learning_rate
        self.robot_environment = robot_environment
        self.robot_arm = robot_arm
        self.action_history = None
        self.state_history = None
        self.network = None
        self.outputLayer = None
        self.fc2 = None
        self.fc1 = None
        self.inputLayer = None

    def constructNN(self):
        self.inputLayer = InputLayer(input_shape=(1,2))
        self.fc1 = Dense(units=self.fc1_dims, activation='relu')
        self.fc2 = Dense(units=self.fc2_dims, activation='relu')
        self.outputLayer = Dense(units=self.output_dims, activation='relu')

        self.network = keras.models.Sequential([
            self.inputLayer,
            self.fc1,
            self.outputLayer
        ])

        self.state_history = [self.robot_environment.start]
        self.action_history = [self.robot_environment.start]

    def getPrevAction(self):
        prevAction = tf.reshape(self.action_history[-1], [1,2])
        return prevAction

    @staticmethod
    def getNextAction(newState):
        mu = newState[0]
        sigma = [[1, 0], [0, 1]]
        actionDist = tfd.MultivariateNormalFullCovariance(loc=mu, covariance_matrix=sigma)
        nextAction, logPdf = actionDist.experimental_sample_and_log_prob()

        logPdf = tf.norm(mu)
        return nextAction, logPdf


    def closeToGoal(self, newAction, threshold):
        theta0 = newAction[0]
        theta1 = newAction[1]

        goal = self.robot_environment.goal

        link1_length, link2_length = self.robot_arm.link_lengths[0], self.robot_arm.link_lengths[1]

        x_joint1, y_joint1 = link1_length * tf.math.cos(theta0), link1_length * tf.math.sin(theta0)
        x_joint2, y_joint2 = x_joint1 + link2_length * tf.math.cos(theta1), y_joint1 + link2_length * tf.math.sin(theta1)

        endEffectorPosition = tf.stack(values=[[x_joint2], [y_joint2]], axis=0)

        goalFunction = 1/(1+tf.math.square(tf.norm(endEffectorPosition-goal)))
        goalPotentialFunctionValue = 1000*goalFunction

        isCloseToGoal = (tf.norm(endEffectorPosition - goal) <= threshold)

        return goalPotentialFunctionValue, isCloseToGoal

    def getNextReward(self, newAction):
        goalPotentialFunction, isCloseToGoal = self.closeToGoal(newAction=newAction, threshold=0.3)
        reward = goalPotentialFunction

        reward = tf.constant(reward)
        return reward

    def resetExperiment(self):
        self.state_history = [self.robot_environment.start]
        self.action_history = [self.robot_environment.start]
        self.reward_history = [0]

    def storeTransition(self, newState, newAction, newReward):
        self.state_history.append(newState)
        self.action_history.append(newAction)

    def print(self):
        print(" XXXXXXXXXXXXXXX NETWORK SUMMARY XXXXXXXXXXXXXXX ")
        self.network.summary()
        print(" XXXXXXXXXXXXXXX NETWORK SUMMARY XXXXXXXXXXXXXXX ")

    def returnlogPDFVal(sigma, mean, sampleVal):
        pi = tf.constant(m.pi)
        pdf = tf.math.pow(tf.linalg.det(2 * pi * sigma), -1 / 2) * tf.math.exp(
            -1 / 2 * tf.linalg.matmul(tf.linalg.matmul(mean - sampleVal, tf.linalg.inv(sigma)),
                                      tf.transpose(mean - sampleVal)))
        logPdf = tf.math.log(pdf)

        return logPdf

    @staticmethod
    def computePolicy(newAction, newState):
        pass


def trainNetwork(pgAlgo, epochs, iterations, Arm):
    for i in range(epochs):
        for j in range(iterations):
            prevAction = pgAlgo.getPrevAction()
            with tf.GradientTape() as tape:
                newState = pgAlgo.network.call(prevAction)
                newAction, logPdf = pgAlgo.getNextAction(newState)

                newReward = pgAlgo.getNextReward(newAction)


                pgAlgo.storeTransition(newState=newState, newAction=newAction, newReward=newReward)
                policy_val = pgAlgo.computePolicy(newReward, logPdf)

                grads = tape.gradient(policy_val, pgAlgo.network.trainable_variables)

                pprint("Gradients  =")
                pprint(grads)
                gradDesc = tf.keras.optimizers.SGD(learning_rate=pgAlgo.learning_rate)
                gradDesc.apply_gradients(zip(grads, pgAlgo.network.trainable_variables))

        pgAlgo.resetExperiment()


def testNetwork(pgAlgo, steps, arm):
    s = tuple(pgAlgo.robot_environment.start)
    g = tuple(pgAlgo.robot_environment.goal)

    route = [s]
    roadmap = {s:None}

    for i in range(steps):
        prevPosition = route[-1]
        nextPosition = pgAlgo.network(prevPosition)
        route.append(nextPosition)
        roadmap[nextPosition] = prevPosition

        goalPotentialFunctionValue, isCloseToGoal = pgAlgo.closeToGoal(newAction=nextPosition[0], threshold=0.1)

        if isCloseToGoal:
            break

    animate(arm, roadmap, route, pgAlgo.robot_environment.start, pgAlgo.robot_environment.obstacles)


def main():
    arm = NLinkArm(LINK_LENGTH, [0,0])
    robot_environment = robotEnv(obstacles=tf.reshape(tf.convert_to_tensor(()), (0, 3)),
                                      start=tf.constant([1, 0]),
                                      goal=tf.constant([0.0, 4.0]),
                                      link_length=tf.constant([2, 2]))
    visualize_spaces(arm, [1, 0] , [[-4,-4,0.3]])

    pgAlgo = policyGradientAlgorithm(robot_environment=robot_environment,robot_arm=arm)
    pgAlgo.constructNN()

    trainNetwork(pgAlgo, epochs=1, iterations=3, Arm=arm)


if __name__ == "__main__":
    main()