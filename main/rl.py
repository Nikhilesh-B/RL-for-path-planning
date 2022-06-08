import tensorflow as tf
from keras.layers import Dense, InputLayer, Dropout
from tensorflow import keras
from keras.optimizer_v2.adam import Adam
from nlinkarm import NLinkArm
from helper import visualize_spaces, animate, detect_collision
from pprint import pprint
from constants import OBSTACLES, START, GOAL, LINK_LENGTH
from tensorflow_probability import distributions

tfd = distributions


class policyGradientAlgorithm:
    def __init__(self, learning_rate=0.001, gamma=0.003, horizon=4, 
                inputLayer_dims=2, fc1_dims=2, fc2_dims=10, output_dims=2):
        self.gamma = gamma
        self.horizon = horizon
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.inputLayer_dims = inputLayer_dims
        self.output_dims = output_dims
        self.learning_rate = learning_rate

        self.inputLayer = InputLayer(input_shape=(self.inputLayer_dims,))
        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.output = Dense(self.output_dims, activation='relu')

        self.network = keras.models.Sequential([
            self.inputLayer, 
            self.fc1,
            self.output
        ])

        self.state_history = [START]
        self.action_history = [START]
        self.reward_history = [0] 
        self.discountedRewardHistory = None

    def getPrevAction(self):
        return [self.action_history[-1]]
    
    @staticmethod
    def getNextAction(newState):
        mu = newState[0]
        sigma = [[1, 0], [0, 1]]
        actionDist = tfd.MultivariateNormalFullCovariance(loc=mu, covariance_matrix=sigma)
        nextAction, logPdf = actionDist.experimental_sample_and_log_prob()
        return nextAction, logPdf

    def detect_collision(self, arm, config, obstacles):
        """
        :param obstacles: circular obstacles list of lists
        :param arm: NLinkArm object
        :param config: Configuration (joint angles) of the arm
        :return: True if any part of arm collides with obstacles, False otherwise
        """
        # here we have all this numpy associated stuff that may need to be converted to TF
        arm.update_joints(config)
        points = arm.points
        for k in range(len(points) - 1):
            for circle in obstacles:
                a_vec = tf.Variable(points[k])
                b_vec = tf.Variable(points[k + 1])
                c_vec = tf.Variable([circle[0], circle[1]])
                radius = circle[2]

                line_vec = b_vec - a_vec
                line_mag = tf.norm(line_vec)
                circle_vec = c_vec - a_vec
                proj = (1/line_mag)*tf.tensordot(circle_vec, line_vec)

                if proj <= 0:
                    closest_point = a_vec
                elif proj >= line_mag:
                    closest_point = b_vec
                else:
                    closest_point = a_vec + line_vec * proj / line_mag

                if tf.norm(closest_point - c_vec) <= radius:
                    return True

        return False

    @staticmethod
    def closeToGoal(newAction, Arm, threshold):
        theta0 = newAction[0][0]
        theta1 = newAction[0][1]

        link1_length, link2_length = Arm.link_lengths[0], Arm.link_lengths[1]

        x_joint1, y_joint1 = link1_length*tf.math.cos(theta0), link1_length*tf.math.sin(theta0)
        x_joint2, y_joint2 = x_joint1+link2_length*tf.math.cos(theta1), y_joint1+link2_length*tf.math.sin(theta1)

        endEffectorPosition = tf.concat([x_joint2, y_joint2], axis=0)

        g = tf.Variable(GOAL)
        goalPotentialFunction = 10000/(1+tf.norm(endEffectorPosition-g))

        isCloseToGoal = (tf.norm(endEffectorPosition-g) <= threshold)

        return goalPotentialFunction, isCloseToGoal

    def getNextReward(self, newAction, Arm):
        reward = 0

        if self.detect_collision(arm=Arm, config=newAction, obstacles=OBSTACLES):
            reward -= 1000

        goalPotentialFunction, isCloseToGoal = self.closeToGoal(newAction=newAction, Arm=Arm, threshold=0.3)
        reward += goalPotentialFunction

        reward  = tf.Variable(reward)
        return reward

    def resetExperiment(self):
        self.state_history = [START]
        self.action_history = [START]
        self.reward_history = [0]

    def storeTransition(self, newState, newAction, newReward):
        self.state_history.append(newState)
        self.action_history.append(newAction)
        self.reward_history.append(newReward)

    def getStateHistory(self):
        return self.state_history

    def getActionHistory(self):
        return self.action_history

    def getDiscountedRewardHistory(self):
        return self.discountedRewardHistory

    def print(self):
        print(" XXXXXXXXXXXXXXX ")
        self.network.summary()
        print(" XXXXXXXXXXXXXXX ")

    def computeDiscountedReward(self):
        self.discountedRewardHistory = [0]*len(self.state_history)
        for k in range(0, len(self.reward_history)):
            val = 0
            for t in range(k, len(self.reward_history)):
                val += self.reward_history[t]*(self.gamma ** (t - k))
            
            self.discountedRewardHistory[k] = val

        self.discountedRewardHistory = tf.convert_to_tensor(self.discountedRewardHistory)

        return self.discountedRewardHistory

    @staticmethod
    def computePolicy(reward, logPdf):
        return -1*reward*logPdf


def trainNetwork(pgAlgo, epochs, iterations, Arm):
    for i in range(epochs):
        for j in range(iterations):
            with tf.GradientTape() as tape:
                prevAction = pgAlgo.getPrevAction()
                newState = pgAlgo.network(tf.convert_to_tensor(prevAction))
                newAction, logPdf = pgAlgo.getNextAction(newState)
                newReward = pgAlgo.getNextReward(newState, Arm)

                policy_val = pgAlgo.computePolicy(newReward, logPdf)
                grads = tape.gradient(policy_val, pgAlgo.network.trainable_variables)
                gradDesc = tf.keras.optimizers.SGD(learning_rate=pgAlgo.learning_rate)
                gradDesc.apply_gradients(zip(grads, pgAlgo.network.trainable_variables))

        pgAlgo.resetExperiment()


def testNetwork(pgAlgo, steps, Arm):
    s = tuple(START)
    g = tuple(GOAL)

    route = [s]
    roadmap = {s:None}

    for i in range(steps):
        p = route[-1]
        prevPosition = np.array([np.array(p)])
        nextPosition = pgAlgo.network.predict(tf.convert_to_tensor(prevPosition))
        n = tuple(nextPosition[0])
        route.append(n)
        roadmap[n] = p

        goalPotentialFunctionValue, isCloseToGoal = pgAlgo.closeToGoal(Arm=Arm, newAction=nextPosition[0], threshold=0.1)

        if isCloseToGoal:
            break

    animate(Arm, roadmap, route, START, OBSTACLES)


def main():
    Arm = NLinkArm(LINK_LENGTH, [0,0])
    #visualize_spaces(Arm, START, OBSTACLES)
    
    pgAlgo = policyGradientAlgorithm()

    trainNetwork(pgAlgo, epochs=1, iterations=10, Arm=Arm)
    pgAlgo.print()

    #testNetwork(pgAlgo, steps=10, Arm=Arm)


def main2():
    Arm = NLinkArm(LINK_LENGTH, [0,0])
    visualize_spaces(Arm, START, OBSTACLES)


if __name__ == "__main__":
    main()