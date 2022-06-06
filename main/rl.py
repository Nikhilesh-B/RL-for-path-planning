import tensorflow as tf
from keras.layers import Dense, InputLayer, Dropout
from tensorflow import keras
from keras.optimizer_v2.adam import Adam
from nlinkarm import NLinkArm
from helper import visualize_spaces, animate, detect_collision
from pprint import pprint
import numpy as np
from constants import OBSTACLES, START, GOAL, LINK_LENGTH
from tensorflow_probability import distributions

tfd = distributions


class policyGradientAlgorithm:
    def __init__(self, learning_rate=0.001, gamma=0.003, horizon=4, inputLayer_dims=2, fc1_dims=256, fc2_dims=256, output_dims=2):
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
            self.fc2, 
            self.output
        ])

        self.state_history = [START]
        self.action_history = [START]
        self.reward_history = [0] 
        self.discountedRewardHistory = None

    def getPrevAction(self):
        return [self.action_history[-1]]
    
    def getNextAction(self, newState):
        mu = newState[0]
        sigma = [[1,0],[0,1]]
        actionDist = tfd.MultivariateNormalFullCovariance(loc=mu, covariance_matrix=sigma)
        nextAction, logProb = actionDist.experimental_sample_and_log_prob()
        return nextAction

    def clostToGoal(self, ARM, newAction):
        theta0 = newAction[0]
        theta1 = newAction[1]

        x_joint1, y_joint1 = 0, 0
        x_joint2, y_joint2 = 0, 0

        link1_length, link2_length = ARM.link_lengths[0], ARM.link_lengths[1]

        x_joint1, y_joint1 = link1_length*np.cos(theta0), link1_length*np.sin(theta0)
        x_joint2, y_joint2 = x_joint1+link2_length*np.cos(theta1), y_joint1+link2_length*np.sin(theta1)

        endEffectorPosition = np.array([x_joint2, y_joint2])

        potentialFunction





    def getNextReward(self, newAction, ARM):
        #this next reward should give you higher and higher values if your close to the goal configuration
        reward = 0
        if detect_collision(arm=ARM, config=newAction, OBSTACLES=OBSTACLES):
            reward -= 100

        if self.closeToGoal(ARM, newAction):
            reward += 200

        return reward



    def resetExperiment(self):
        self.state_history = [GOAL]
        self.action_history = [GOAL]
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
        print("XXXXXXXXXXXXXXX")
        print("Network summary")
        self.network.summary()
        print("XXXXXXXXXXXXXXX")



    def computeDiscountedReward(self):
        self.discountedRewardHistory = [0]*len(self.state_history)
        for k in range(0, len(self.reward_history)):
            val = 0
            for t in range(k, len(self.reward_history)):
                val += self.reward_history[t]*(self.gamma ** (t - k))
            
            self.discountedRewardHistory[k] = val

        self.discountedRewardHistory = tf.convert_to_tensor(self.discountedRewardHistory)

        return self.discountedRewardHistory

    def computePolicy(self, action, state, discountedReward):
        probDist = distributions.MultivariateNormalFullCovariance(loc=state, covariance_matrix=[[1,0],[0,1]])
        logPdf = tf.math.log(probDist.prob(action))
    
        return discountedReward*logPdf


            

def trainNetwork(pgAlgo, epochs, iterations):
    for i in range(epochs):
        for j in range(iterations):
            with tf.GradientTape() as tape:
                prevAction = pgAlgo.getPrevAction()
                newState = pgAlgo.network(tf.convert_to_tensor(prevAction))
                newAction = pgAlgo.getNextAction(newState)
                newReward = pgAlgo.getNextReward(newAction)

                policy_val = pgAlgo.computePolicy(newState, newAction, newReward)
                grads = tape.gradient(policy_val, pgAlgo.network.trainable_variables)
                adam = Adam(learning_rate=pgAlgo.learning_rate)
                adam.apply_gradients(zip(grads, pgAlgo.network.trainable_variables))
        '''
        pgAlgo.computeDiscountedReward()
        stateHistory = pgAlgo.getStateHistory()
        actionHistory = pgAlgo.getActionHistory()
        discountedRewardsHistory = pgAlgo.getDiscountedRewardHistory().numpy()

       
        for k in range(len(stateHistory)):
            state = stateHistory[k]
            action = actionHistory[k]
            discountedRewardsHistory = discountedRewardsHistory[k]
            policy_val = pgAlgo.computePolicy(state, action, discountedRewardsHistory)
            grads = tape.gradient(policy_val, pgAlgo.network.trainable_variables)
            adam = Adam(learning_rate=pgAlgo.learning_rate)
            adam.apply_gradients(zip(grads, pgAlgo.network.trainable_variables))
        '''
        pgAlgo.resetExperiment()


def testNetwork(pgAlgo, steps, ARM):
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

        if np.linalg.norm(np.array(g) - np.array(n)) < 0.2:
            break

    animate(ARM, roadmap, route, START, OBSTACLES)




def main():
    ARM = NLinkArm(LINK_LENGTH, [0,0])
    visualize_spaces(ARM, START, OBSTACLES)

    roadmap = {(1.0,0.0):None,
               (1.1, 0.0):(1.0,0.0),
               (1.2, 0.0):(1.1, 0.0),
               (1.3, 0.0):(1.2, 0.0)}

    route = [(1.0,0.0),(1.1,0.0),(1.2,0.0),(1.3,0.0)]

    animate(ARM, roadmap, route, START, OBSTACLES)

    '''
    pgAlgo = policyGradientAlgorithm()


    trainNetwork(pgAlgo, epochs=10, iterations=10)
    pgAlgo.print()


    testNetwork(pgAlgo, steps=10, ARM=ARM)
    '''
    
    

if __name__ == "__main__":
    main()