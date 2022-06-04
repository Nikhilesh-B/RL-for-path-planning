import tensorflow as tf
from keras.layers import Dense, InputLayer, Dropout
from tensorflow import keras
from keras.optimizer_v2.adam import Adam
from nlinkarm import NLinkArm
from helper import visualize_spaces, animate
import numpy as np
from pprint import pprint 
from scipy.integrate import dblquad
from constants import OBSTACLES, START, GOAL, LINK_LENGTH
from tensorflow_probability import distributions

tfd = distributions




class policyGradientAlgorithm:
    def __init__(self, learning_rate=0.001 ,gamma=0.003, horizon=4, inputLayer_dims=2, fc1_dims=256, fc2_dims=256, output_dims=2):
        self.gamma = gamma
        self.horizon = horizon
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.inputLayer_dims = inputLayer_dims
        self.output_dims = output_dims
        self.learning_rate = learning_rate

        self.inputLayer = InputLayer(self.inputLayer_dims)
        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.output = Dense(self.output_dims, activation='relu')

        self.network = keras.models.Sequential([
            self.inputLayer, 
            self.fc1,
            self.fc2, 
            self.output
        ])

        self.state_history = [GOAL] 
        self.action_history = [GOAL] 
        self.reward_history = [0] 
        self.discountedRewardHistory = None

    def getPrevAction(self):
        return self.action_history[-1]
    
    def getNextAction(self, newState):
        mu = newState[0]
        sigma = [[1,0],[0,1]]
        actionDist = tfd.MultivariateNormalFullCovariance(loc=mu, covariance_matrix=sigma)
        nextAction, logProb = actionDist.experimental_sample_and_log_prob()
        return nextAction

    def getNextState(self, prevAction):
        return self.network.predict(np.array([prevAction]))

    def getNextReward(self, newAction):
        reward = 0
        for obst in OBSTACLES:
            obstPotentialFunction = tfd.MultivariateNormalFullCovariance(loc=obst[:-1],covariance_matrix=[[1,0],[0,1]])
            reward -= obstPotentialFunction.prob(newAction)

        
        goalPotentialFunction  = tfd.MultivariateNormalFullCovariance(loc=GOAL, covariance_matrix=[[1,0],[0,1]])
        reward += goalPotentialFunction.prob(newAction)

        return reward

    def resetExperiment(self):
        self.state_history = []
        self.action_history = []
        self.reward_history = []


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
        print("---------------")
        print("Action history", self.action_history)
        print("XXXXXXXXXXXXXXX")
        print("---------------")
        print("State history", self.state_history)
        print("XXXXXXXXXXXXXXXX")



    def computeDiscountedReward(self):
        self.discountedRewardHistory = np.zeros(len(self.state_history))
        for k in range(0, len(self.reward_history)):
            val = 0
            for t in range(k, len(self.reward_history)):
                val += self.reward_history[t]*((self.gamma)**(t-k))
            
            self.discountedRewardHistory[k] = val
        
        return self.discountedRewardHistory

            #we have the action that we have taken 

    def computeLoss(self, action, state, discountedReward):
        probDist = distributions.MultivariateNormalFullCovariance(loc=state, covariance_matrix=[[1,0],[0,1]])
        logPdf = tf.math.log(probDist.prob(action))
    
        return -discountedReward*logPdf


            

def trainNetwork(pgAlgo, epochs, iterations):

    for i in range(epochs):
        for j in range(iterations):
            prevAction = pgAlgo.getPrevAction()
            newState = pgAlgo.getNextState(prevAction)
            newAction  = pgAlgo.getNextAction(newState)
            newReward  = pgAlgo.getNextReward(newAction)
            
            pgAlgo.storeTransition(newState, newAction, newReward)

        pgAlgo.computeDiscountedReward()

        
        stateHistory = pgAlgo.getStateHistory()
        actionHistory = pgAlgo.getActionHistory()
        discountedRewardsHistory = pgAlgo.getDiscountedRewardHistory()

        print("XXXXXXXXXXXXXXXXXXXXXXXX")
        print("XXXXXXXXXXXXXXXXXXXXXXXX")
        print("XXXXXXXXXXXXXXXXXXXXXXXX")
        print("XXXXXXXXXXXXXXXXXXXXXXXX")
        print("XXXXXXXXXXXXXXXXXXXXXXXX")
        print("State History")
        pprint(stateHistory)
        print("XXXXXXXXXXXXXXXXXXXXXXXX")
        print("XXXXXXXXXXXXXXXXXXXXXXXX")
        print("XXXXXXXXXXXXXXXXXXXXXXXX")
        print("XXXXXXXXXXXXXXXXXXXXXXXX")
        print("XXXXXXXXXXXXXXXXXXXXXXXX")
        print("XXXXXXXXXXXXXXXXXXXXXXXX")
        print("XXXXXXXXXXXXXXXXXXXXXXXX")
        print("XXXXXXXXXXXXXXXXXXXXXXXX")
        print("XXXXXXXXXXXXXXXXXXXXXXXX")
        print("XXXXXXXXXXXXXXXXXXXXXXXX")
        print("Action History")
        pprint(actionHistory)
        print("XXXXXXXXXXXXXXXXXXXXXXXX")
        print("XXXXXXXXXXXXXXXXXXXXXXXX")
        print("XXXXXXXXXXXXXXXXXXXXXXXX")
        print("XXXXXXXXXXXXXXXXXXXXXXXX")
        print("XXXXXXXXXXXXXXXXXXXXXXXX")
        print("XXXXXXXXXXXXXXXXXXXXXXXX")
        print("XXXXXXXXXXXXXXXXXXXXXXXX")
        print("XXXXXXXXXXXXXXXXXXXXXXXX")
        print("XXXXXXXXXXXXXXXXXXXXXXXX")
        print("XXXXXXXXXXXXXXXXXXXXXXXX")
        print("Discounted Reward History")
        pprint(discountedRewardsHistory)
        print("XXXXXXXXXXXXXXXXXXXXXXXX")
        print("XXXXXXXXXXXXXXXXXXXXXXXX")
        print("XXXXXXXXXXXXXXXXXXXXXXXX")
        print("XXXXXXXXXXXXXXXXXXXXXXXX")
        print("XXXXXXXXXXXXXXXXXXXXXXXX")


        for k in range(len(stateHistory)):
            state, action, discountedRewardsHistory = stateHistory[k], actionHistory[k], discountedRewardsHistory[k]


            with tf.GradientTape() as tape:
                loss_val = pgAlgo.computeLoss(state, action, discountedRewardsHistory)
                grads = tape.gradient(loss_val, pgAlgo.network.trainable_variables)
                adam = Adam(learning_rate=pgAlgo.learning_rate)
                adam.apply_gradients(zip(grads,pgAlgo.network.trainable_variables))

        pgAlgo.resetExperiment()
    



def main():
    ARM = NLinkArm(LINK_LENGTH, [0,0])
    #visualize_spaces(ARM, START, OBSTACLES)

    pgAlgo = policyGradientAlgorithm()


    trainNetwork(pgAlgo, epochs=10, iterations=10)


    #pgAlgo.print()
    '''roadmap = {(1.0,0.0):None,
               (0.83, 0.29):(1.0,0.0),
               (0.62, 0.53):(0.83, 0.29),
               (1.0,0.5):(1.33,0.52)}'''

    #route = [(1.0,0.0),(0.83,0.29),(0.62,0.53),(1.33,0.53),(1.0,0.5)]

   #animate(ARM, roadmap, route, START, OBSTACLES) 

if __name__ == "__main__":
    main()