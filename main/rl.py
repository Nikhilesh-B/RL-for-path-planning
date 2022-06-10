import tensorflow as tf
from nlinkarm import NLinkArm
from helper import visualize_spaces, animate, detect_collision
from constants import OBSTACLES, START, GOAL, LINK_LENGTH
from tensorflow_probability import distributions
from keras.layers import Dense, InputLayer
from tensorflow import keras
from robotEnv import robotEnv

tfd=distributions

class policyGradientAlgorithm():
    def __init__(self, robot_environment, robot_arm, learning_rate=0.001, gamma=0.003, horizon=4, inputLayer_dims=2, fc1_dims=1, fc2_dims=1,
                 output_dims=1):
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
        self.output = Dense(units=self.output_dims, activation='relu')

        self.network = keras.models.Sequential([
            self.inputLayer,
            self.fc1,
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
        mu = newState[0]
        sigma = [[1, 0], [0, 1]]
        actionDist = tfd.MultivariateNormalFullCovariance(loc=mu, covariance_matrix=sigma)
        nextAction, logPdf = actionDist.experimental_sample_and_log_prob()
        return nextAction, logPdf

    def detect_collision(self, arm, config, obstacles):
        arm.update_joints(config)
        points = arm.points
        for k in range(len(points) - 1):
            print("Obstacles=",obstacles)
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
        print("New action=",newAction)
        theta0 = newAction[0]
        theta1 = newAction[1]

        link1_length, link2_length = self.robot_arm.link_lengths[0], self.robot_arm.link_lengths[1]

        x_joint1, y_joint1 = link1_length * tf.math.cos(theta0), link1_length * tf.math.sin(theta0)
        x_joint2, y_joint2 = x_joint1 + link2_length * tf.math.cos(theta1), y_joint1 + link2_length * tf.math.sin(theta1)

        endEffectorPosition = tf.concat([x_joint2, y_joint2], axis=0)

        g = self.robot_environment.goal
        goalFunction = tfd.MultivariateNormalFullCovariance(loc=endEffectorPosition, covariance_matrix=[[1, 0], [0, 1]])
        goalPotentialFunctionValue = 1000 * goalFunction.prob(newAction)

        isCloseToGoal = (tf.norm(endEffectorPosition - g) <= threshold)

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

    @staticmethod
    def computePolicy(reward, logPdf):
        return reward * logPdf


def trainNetwork(pgAlgo, epochs, iterations, Arm):
    for i in range(epochs):
        for j in range(iterations):
            prevAction = pgAlgo.getPrevAction()
            print("Here is the previous action that goes into the input = ",prevAction)
            with tf.GradientTape() as tape:
                newState = pgAlgo.network.call(prevAction)
                newAction, logPdf = pgAlgo.getNextAction(newState)
                newReward = pgAlgo.getNextReward(newAction)
                pgAlgo.storeTransition(newState=newState, newAction=newAction, newReward=newReward)

                policy_val = pgAlgo.computePolicy(newReward, logPdf)
                grads = tape.gradient(policy_val, pgAlgo.network.trainable_variables)
                gradDesc = tf.keras.optimizers.SGD(learning_rate=pgAlgo.learning_rate)
                gradDesc.apply_gradients(zip(grads, pgAlgo.network.trainable_variables))

        pgAlgo.resetExperiment()


def testNetwork(pgAlgo, steps, arm, robot_environment):
    s = tuple(robot_environment.start)
    g = tuple(robot_environment.goal)

    route = [s]
    roadmap = {s:None}

    for i in range(steps):
        p = route[-1]
        prevPosition = np.array([np.array(p)])
        nextPosition = pgAlgo.network.predict(tf.convert_to_tensor(prevPosition))
        n = tuple(nextPosition[0])
        route.append(n)
        roadmap[n] = p

        goalPotentialFunctionValue, isCloseToGoal = pgAlgo.closeToGoal(newAction=nextPosition[0], threshold=0.1)

        if isCloseToGoal:
            break

    animate(arm, roadmap, route, pgAlgo.robot_environment.start, pgAlgo.robot_environment.obstacles)


def main():
    arm = NLinkArm(LINK_LENGTH, [0,0])
    robot_environment = robotEnv(obstacles= tf.reshape(tf.convert_to_tensor(()), (0, 3)),
                                      start=tf.constant([1, 0]),
                                      goal=tf.constant([2.16, 3.36]),
                                      link_length=tf.constant([2, 2]))

    pgAlgo = policyGradientAlgorithm(robot_environment=robot_environment,robot_arm=arm)
    pgAlgo.constructNN()

    trainNetwork(pgAlgo, epochs=1, iterations=10, Arm=arm)
    pgAlgo.print()


if __name__ == "__main__":
    main()