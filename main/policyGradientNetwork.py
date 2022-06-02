import keras
from keras.layers import Dense, InputLayer
from constants import GOAL, LINK_LENGTH, OBSTACLES
from scipy.stats import multivariate_normal
import numpy as np



class policyGradientNetwork(keras.Model):
    def __init__(self, inputLayer_dims=2, fc1_dims=256, fc2_dims=256, output_dims=2):
        super(policyGradientNetwork, self).__init__()
        self.inputLayer_dims = inputLayer_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.output_dims = output_dims

        self.inputLayer = InputLayer(shape=(self.inputLayer_dims,))
        self.fc1 = Dense(self.fc1, activation='relu')
        self.fc2 = Dense(self.fc2, activation='relu')
        self.output = Dense(self.output_dims, activation='relu')

        self.network = keras.Model(inputs=self.inputLayer, outputs=self.output)



    def compute_loss(self, action, reward):
        mu = self.network.predict(action)

        dist = multivariate_normal(mean=mu,cov=np.eye(2))
        prob = dist.pdf(action)
        



        return 


    def forwardPass(self, state):
        return self.output(
                            self.fc1(
                                    self.fc2(
                                            self.inputLayer(state)
                                            )
                                    )
                            )
    