import tensorflow as tf
from keras.layers import Dense, InputLayer, Dropout
import keras
from nlinkarm import NLinkArm
from helper import visualize_spaces, animate
import numpy as np
from scipy.stats import multivariate_normal 
from pprint import pprint 
from scipy.integrate import dblquad
from constants import OBSTACLES, START, GOAL, LINK_LENGTH


class policyGradientAlgorithm:
    def __init__(self, gamma=0.003, horizon=4, inputLayer_dims=2, fc1_dims=256, fc2_dims=256, output_dims=2):
        self.gamma = gamma
        self.horizon = horizon
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.inputLayer_dims = inputLayer_dims
        self.output_dims = output_dims

        self.inputLayer = InputLayer(shape=(self.inputLayer_dims,))
        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.output = Dense(self.output_dims, activation='relu')

        self.network = keras.models.Sequential([
            self.inputLayer, 
            self.fc1,
            self.fc2, 
            self.output
        ])

        self.state_history = [] 
        self.action_history = [] 
        self.reward_history = [] 


    def nextAction(self, newState):
        mu = newState
        sigma = np.eye(2)
        nextAction = np.random.multivariate_normal(mean=mu, cov=sigma)
        return nextAction

    def nextState(self, prevAction):
        return self.network.predict(prevAction)

    def computeReward(self, newAction):
        reward = 0
        for obst in OBSTACLES:
            obstPotentialFunction = multivariate_normal(mean=obst[:-1])
            reward -= obstPotentialFunction.pdf(newAction)
            
        reward += multivariate_normal(mean=GOAL)

        return reward
        


    def store_transition(self, newState, newAction, newReward):
        self.state_history.append(newState)
        self.action_history.append(newAction)
        self.reward_history.append(newReward)


    def computeDiscountedReward(self):
        discountedRewardCalc = np.zeros(len(self.state_history))
        for k in range(0, len(self.reward_history)):
            val = 0
            for t in range(k, len(self.reward_history)):
                val += self.reward_history[t]*((self.gamma)**(t-k))
            
            discountedRewardCalc[k] = val
        
        return discountedRewardCalc

            #we have the action that we have taken 

    def computeLoss(self, action, discountedReward):
        mu  = self.network.predict(action)

        probDist = multivariate_normal(mean=mu, cov=np.eye(2))
        newAction  = np.random.multivariate_normal(mean=mu, cov=np.eye(2))
        pdf_value = probDist
            


def main():
    ARM = NLinkArm(LINK_LENGTH, [0,0])
    visualize_spaces(ARM, START, OBSTACLES)


    pgAlgo = policyGradientAlgorithm()


    roadmap = {(1.0,0.0):None,
               (0.83, 0.29):(1.0,0.0),
               (0.62, 0.53):(0.83, 0.29),
               (1.0,0.5):(1.33,0.52)}

    route = [(1.0,0.0),(0.83,0.29),(0.62,0.53),(1.33,0.53),(1.0,0.5)]

    animate(ARM, roadmap, route, START, OBSTACLES) 

if __name__ == "__main__":
    main()