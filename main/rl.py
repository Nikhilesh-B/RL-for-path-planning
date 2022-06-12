import tensorflow as tf
from nlinkarm import NLinkArm
from helper import visualize_spaces, animate, detect_collision
from constants import OBSTACLES, START, GOAL, LINK_LENGTH
from tensorflow_probability import distributions
from keras.layers import Dense, InputLayer
from tensorflow import keras
from robotEnv import robotEnv
from pprint import pprint
import math as m
import tensorflow as tf

tfd=distributions

class policyGradientAlgorithm():
    def __init__(self, robot_environment, robot_arm, learning_rate=0.001, gamma=0.003, horizon=4, inputLayer_dims=2, fc1_dims=1, fc2_dims=1, output_dims=2):
        self.gamma = gamma
        self.horizon = horizon
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.inputLayer_dims = inputLayer_dims
        self.output_dims = output_dims
        self.learning_rate = learning_rate
        self.robot_environment = robot_environment
        self.robot_arm = robot_arm
        self.discountedRewardHistory = None
        self.reward_history = None
        self.action_history = None
        self.state_history = None
        self.network = None
        self.output = None
        self.fc2 = None
        self.fc1 = None
        self.inputLayer = None


    def constructNN(self):
        self.inputLayer = InputLayer(input_shape=(1,2))
        self.fc1 = Dense(units=self.fc1_dims, activation='relu')
        self.fc2 = Dense(units=self.fc2_dims, activation='relu')
        self.output = Dense(units=self.output_dims, activation='relu')

        self.network = keras.models.Sequential([
            self.inputLayer,
            self.fc1,
            #self.fc2,
            self.output
        ])

        self.state_history = [self.robot_environment.start]
        self.action_history = [self.robot_environment.start]
        self.reward_history = [0]
        self.discountedRewardHistory = None

    def getPrevAction(self):
        prevAction = tf.reshape(self.action_history[-1], [1,2])
        return prevAction

    @staticmethod
    def getNextAction(newState):
        #is the logPdf a function of the mu? at all or is not even?
        mu = newState[0]
        sigma = [[1, 0], [0, 1]]
        actionDist = tfd.MultivariateNormalFullCovariance(loc=mu, covariance_matrix=sigma)
        nextAction, logPdf = actionDist.experimental_sample_and_log_prob()

        #the issue is that this newAction and the logProb of the draw is not at all a function of the original neural network
        logPdf = tf.norm(mu)
        return nextAction, logPdf

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
        reward = 0

        if self.detect_collision(arm=self.robot_arm, config=newAction, obstacles=self.robot_environment.obstacles):
            reward -= 1000

        goalPotentialFunction, isCloseToGoal = self.closeToGoal(newAction=newAction, threshold=0.3)
        reward += goalPotentialFunction

        reward = tf.constant(reward)
        return reward

    def resetExperiment(self):
        self.state_history = [self.robot_environment.start]
        self.action_history = [self.robot_environment.start]
        self.reward_history = [0]

    def storeTransition(self, newState, newAction, newReward):
        self.state_history.append(newState)
        self.action_history.append(newAction)
        self.reward_history.append(newReward)

    def print(self):
        print(" XXXXXXXXXXXXXXX ")
        self.network.summary()
        print(" XXXXXXXXXXXXXXX ")

    def computeDiscountedReward(self):
        self.discountedRewardHistory = [0] * len(self.state_history)
        for k in range(0, len(self.reward_history)):
            val = 0
            for t in range(k, len(self.reward_history)):
                val += self.reward_history[t] * (self.gamma ** (t - k))

            self.discountedRewardHistory[k] = val

        self.discountedRewardHistory = tf.convert_to_tensor(self.discountedRewardHistory)

        return self.discountedRewardHistory

    def returnlogPDFVal(sigma, mean, sampleVal):
        pi = tf.constant(m.pi)
        pdf = tf.math.pow(tf.linalg.det(2 * pi * sigma), -1 / 2) * tf.math.exp(
            -1 / 2 * tf.linalg.matmul(tf.linalg.matmul(mean - sampleVal, tf.linalg.inv(sigma)),
                                      tf.transpose(mean - sampleVal)))
        print("pdf=", pdf)
        logPdf = tf.math.log(pdf)

        return logPdf

    @staticmethod
    def computeLogPdf(newAction, newState):
        '''actually substitute the value and return the next value '''

        #logPdf has no valid gradients that can be applied
        #reward has all the gradients evaluate to nones.





        policy_val = tf.reshape(-reward,shape=(1,1))
        return policy_val


def trainNetwork(pgAlgo, epochs, iterations, Arm):
    for i in range(epochs):
        for j in range(iterations):
            prevAction = pgAlgo.getPrevAction()
            with tf.GradientTape() as tape:
                newState = pgAlgo.network.call(prevAction)
                newAction, logPdf = pgAlgo.getNextAction(newState)

                print("Here are the values of the weights in the network")
                pprint(pgAlgo.network.get_weights())

                pprint("Hey here's the new state=")
                pprint(newState)

                pprint("Hey here's the new action=")
                pprint(newAction)

                newReward = pgAlgo.getNextReward(newAction)

                pprint("Here's the associated reward=")
                pprint(newReward)
                pgAlgo.storeTransition(newState=newState, newAction=newAction, newReward=newReward)
                policy_val = pgAlgo.computePolicy(newReward, logPdf)
                pprint("Here's the new policy value=")
                pprint(policy_val)
                #the gradients somehow all evaluate to zero so what's the issue here
                grads = tape.gradient(policy_val, pgAlgo.network.trainable_variables)
                #here we have the grads that we can manipulate and work with directly.


                grads = newReward*tape.gradient(policy_val)
                
                

                pprint("Gradients=")
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
    pgAlgo.print()


if __name__ == "__main__":
    main()