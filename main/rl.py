import tensorflow as tf

from policyGradientNetwork import policyGradientNetwork
from turtle import position
from nlinkarm import NLinkArm
from helper import visualize_spaces, animate
import numpy as np
from scipy.stats import multivariate_normal 
from pprint import pprint 
from scipy.integrate import dblquad
from constants import OBSTACLES, START, GOAL, LINK_LENGTH


class arm:
    def __init__(self, gamma=0.003, horizon=4, fc1_dims = 256, fc2_dims=256):
        self.gamma = gamma
        self.horizon = horizon
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.policyNet = policyGradientNetwork()

        self.state_history = [] #predicted mean and the covariance for the continous action
        self.action_history = [] #actual positions we moved to
        self.reward_history = [] 


    def nextAction(self, newState):
        mu = newState
        sigma = np.eye(2)
        nextAction = np.random.multivariate_normal(mean=mu, cov=sigma)


        return nextAction

    def store_transition(self, newState, newAction, newReward):
        self.state_history.append(newState)
        self.action_history.append(newAction)
        self.action_history.append(newReward)

    def learn(self):
        























def computeExpectedValueIntegral(mean, variance):
    pass



def computedExpectedValueIntegral(mu, sigma, theta_0, theta_1):
    



    def returnIntegralFunctionValue(theta_0, theta_1):
        position=np.array([theta_0,theta_1])
        attractivePotentialFunction = multivariate_normal(mean=GOAL, cov=np.eye(2))
        value = attractivePotentialFunction.pdf(position)        

        for obst in OBSTACLES:
            repulsivePotentialFunction = multivariate_normal(mean=obst, cov=np.eye(2))
            value = value - repulsivePotentialFunction.pdf(position)
        
        underlyingDistribution = multivariate_normal(mean=position,cov=np.eye(2))
        value = value*underlyingDistribution.pdf(position)


        return value 

def main():
    ARM = NLinkArm(LINK_LENGTH, [0,0])
    visualize_spaces(ARM, START, OBSTACLES)

    roadmap = {(1.0,0.0):None,
               (0.83, 0.29):(1.0,0.0),
               (0.62, 0.53):(0.83, 0.29),
               (1.0,0.5):(1.33,0.52)}

    route = [(1.0,0.0),(0.83,0.29),(0.62,0.53),(1.33,0.53),(1.0,0.5)]

    animate(ARM, roadmap, route, START, OBSTACLES) 

if __name__ == "__main__":
    main()