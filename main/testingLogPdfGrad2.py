import keras.models
import tensorflow as tf
import math as m
from keras.layers import Dense, InputLayer
from tensorflow_probability import distributions as tfd

def getNextAction(newState):
    mu = newState[0]
    sigma = [[1, 0], [0, 1]]
    actionDist = tfd.MultivariateNormalFullCovariance(loc=mu, covariance_matrix=sigma)
    nextAction, logPdf = actionDist.experimental_sample_and_log_prob()

    logPdf = tf.norm(mu)
    return nextAction, logPdf


def returnlogPDFVal(sigma, mean, sampleVal):
    pi = tf.constant(m.pi)
    pdf = tf.math.pow(tf.linalg.det(2 * pi * sigma), -1 / 2) * tf.math.exp(
        -1 / 2 * tf.linalg.matmul(tf.linalg.matmul(mean - sampleVal, tf.linalg.inv(sigma)),
                                  tf.transpose(mean - sampleVal)))
    print("pdf=", pdf)
    logPdf = tf.math.log(pdf)
    print("log pdf=",logPdf)
    return logPdf


inputLayer = InputLayer(input_shape=(1,2))
outputLayer = Dense(units=2, activation='relu' )

network = keras.models.Sequential([
    inputLayer,
    outputLayer
])

input = tf.constant([[1, 2]])




with tf.GradientTape() as tape:
    #here we have to calculate the gradient of the logPdfValue with respect to the trainable variables.
    newState = network.call(input)
    print("New state=", newState)
    nextAction, pdfUnworkable = getNextAction(newState)
    print("Next action=", nextAction)
    logPdfVal = returnlogPDFVal([[1.0, 0.0], [0.0, 1.0]], newState, nextAction)

    grads = tape.gradient(logPdfVal, network.trainable_variables)
    print("Grads=",grads)
    print("1st gradient=",grads[0][0][0])
    gradDesc = tf.keras.optimizers.SGD(learning_rate=0.1)
    gradDesc.apply_gradients(zip(grads,network.trainable_variables))











