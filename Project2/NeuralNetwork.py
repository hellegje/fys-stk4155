##FFNN
from typing import List
from enum import Enum
from random import random, seed
import numpy as np
import sklearn.metrics as error

class ActivationFunction(Enum):
    Sigmoid = 0,
    RELU = 1,
    LeakyRELU = 2

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z)) 

def sigmoid_derivative(a):
    return a * (1 - a)

#TODO: Check these
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return np.where(z > 0, 1, 0)

def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

def leaky_relu_derivative(z, alpha=0.01):
    return np.where(z > 0, 1, alpha)

class CostFunction(Enum):
    MSE = 0,
    R2 = 1

def mse(y_target, y_pred):
    n = np.size(y_pred)  # Number of data points
    return np.sum((y_target - y_pred)**2)/n

def r2(y_target, y_pred):
    return error.r2_score(y_target, y_pred)

class NeuralNetwork:
    def __init__(self, 
                 #output_layer, 
                 cost_function, 
                 random_state = None,
                 #Output layer:
                 outputlayer_activation = None, 
                 n_output_neurons = 1, 
                 n_outputs = 1, 
                 #Hidden layer:
                 hidden_activation = ActivationFunction.Sigmoid,
                 hidden_features = 1,
                 hidden_neurons = 1,
                 hidden_layers = None):
        #self.input_layer = input_layer
        self.hidden_layers = hidden_layers
        self.cost_function = cost_function
        output_layer = OutputLayer(outputlayer_activation, n_outputs, random_state)

        #User can self-define a list of hidden layers, otherwise the default is one layer with given values
        if hidden_layers == None:
            layer = HiddenLayer(hidden_activation, hidden_features, hidden_neurons, random_state)
            self.hidden_layers = list([layer])
        
        n_neurons_last_hiddenlayer = self.hidden_layers[-1]._get_hidden_neurons()
        output_layer._set_weights(n_neurons_last_hiddenlayer)
        self.output_layer = output_layer
            
    
    #So as to be flexible, we can expand the model after creation - will probably need a method like GetHiddenLayers
    def add_hidden_layer(self, position, layer):
        self.hidden_layers.append(layer) #TODO: At a specific position
            
    def evaluate_model(self, predicted_value, target_value):
        match self.cost_function:
            case CostFunction.MSE:
                return mse(target_value, predicted_value)
            case CostFunction.R2:
                return r2(target_value, predicted_value)
            case _:
                raise NotImplementedError(f"Cost function {self.cost_function} not implemented.")

    def FeedForward(self, x):
        activations = []
        a = x  # Initial input is x
        for i, layer in enumerate(self.hidden_layers):
            z = np.matmul(a, layer.get_weights()) + layer.get_biases()
            a = layer.activation(z)
            activations.append(a)  # Collect activation from each layer for backprop
        # Output layer computation (no activation here for regression)
        z_out = np.matmul(a, self.output_layer.get_weights()) + self.output_layer.get_biases()
        activations.append(z_out)
        return activations
    
    def BackPropagation(self, activations, y, x):
        # Initializing gradients
        output_weights_gradient = []
        output_bias_gradient = []
        
        hidden_weights_gradient = []
        hidden_bias_gradient = []
        
        a_output = activations[-1]
        delta_output = (a_output - y) * self.output_layer.derivative(a_output)

        #print(0.5 * ((a_output - y) ** 2))  # MSE for linear output
        
        # Backpropagate through hidden layers
        delta = np.matmul(delta_output, self.output_layer.get_weights().T) * self.hidden_layers[-1].derivative(activations[-2])
        output_weights_gradient = np.matmul(activations[-2].T, delta_output)
        output_bias_gradient = np.sum(delta_output, axis=0, keepdims=True)

        for i in reversed(range(len(self.hidden_layers))):
            # Insert at start of list because working backwards
            hidden_weights_gradient.insert(0, np.matmul(activations[i-1].T, delta) if i > 0 else np.matmul(x.T, delta))
            hidden_bias_gradient.insert(0, np.sum(delta, axis=0, keepdims=True))

            # Prepare delta for the next layer back
            if i > 0:
                delta = np.matmul(delta, self.hidden_layers[i].get_weights().T) * self.hidden_layers[i-1].derivative(activations[i-1])

        
        return output_weights_gradient, output_bias_gradient, hidden_weights_gradient, hidden_bias_gradient
    
    # Update weights and biases using gradients
    def update_parameters(self, gradients, learning_rate):
        output_weights_gradient, output_bias_gradient, hidden_weights_gradient, hidden_bias_gradient = gradients
        
        # Update output layer weights and biases
        self.output_layer.w -= learning_rate * output_weights_gradient
        self.output_layer.b -= learning_rate * output_bias_gradient.squeeze()
        
        # Update hidden layer weights and biases
        for i, layer in enumerate(self.hidden_layers):
            layer.w -= learning_rate * hidden_weights_gradient[i]
            layer.b -= learning_rate * hidden_bias_gradient[i].squeeze()

    
    # Train the model
    def train(self, x, y, epochs, learning_rate, verbosity=False):
        final_loss = 0
        #Becomes relevant if adding a stop condition:
        final_epoch = 0
        for epoch in range(epochs):
            final_epoch = epoch + 1
            # Feedforward pass
            activations = self.FeedForward(x)
            # Compute loss for monitoring (optional)
            loss = self.evaluate_model(activations[-1], y)
            final_loss = loss
            if verbosity:
                print(f"Epoch {epoch+1}, Loss: {loss}")
            # Backpropagation
            gradients = self.BackPropagation(activations, y, x)
            # Update weights and biases
            self.update_parameters(gradients, learning_rate)

        return [final_epoch, final_loss]
    
class Layer:
    def __init__(self, activation_function, n_features, n_hidden_neurons, random_state=None):
        # weights and bias in the hidden layer
        if random_state is not None and random_state >= 0:
            np.random.seed(random_state)
        self.w = np.random.randn(n_features, n_hidden_neurons)
        #self.b = np.zeros(n_hidden_neurons) #+ 0.01 
        self.b = np.zeros((1, n_hidden_neurons))
        #TODO: For different activation functions, a small initial value may be beneficial, especially for relu to avoid vanishing neurons
        self.activation_fnc = activation_function
        self.n_neurons = n_hidden_neurons

    def get_weights(self):
        return self.w
    
    def get_biases(self):
        return self.b
    
    def activation(self, z):
        match self.activation_fnc:
            case ActivationFunction.Sigmoid:
                return sigmoid(z)
            case ActivationFunction.RELU:
                return relu(z)
            case ActivationFunction.LeakyRELU:
                return leaky_relu(z)
            case None:
                return z # Typical for regression type problems
            case _:
                raise NotImplementedError(f"Activation function not implemented for: {self.activation_fnc}")

    def derivative(self, a):
        match self.activation_fnc:
            case ActivationFunction.Sigmoid:
                return sigmoid_derivative(a)
            case ActivationFunction.RELU:
                return relu_derivative(a)
            case ActivationFunction.LeakyRELU:
                return leaky_relu_derivative(a)
            case None:
                return np.ones_like(a) #Linear #TODO: Is this right?
            case _:
                raise NotImplementedError(f"Derivative function not implemented for {self.activation_fnc}")

class HiddenLayer(Layer):
    pass
    def _get_hidden_neurons(self):
        return self.n_neurons

class OutputLayer(Layer):
    def __init__(self, activation_function, n_outputs, random_state=None):
        # weights and bias in the output layer
        #self.b = np.zeros(n_outputs) #+ 0.01 
        self.random_state = random_state
        self.n_outputs = n_outputs
        self.b = np.zeros((1, n_outputs))
        self.activation_fnc = activation_function

    def _set_weights(self, n_hidden_neurons):
        if self.random_state is not None and self.random_state >= 0:
            np.random.seed(self.random_state)
        self.w = np.random.randn(n_hidden_neurons, self.n_outputs)

#TODO: Input layer if/when needed
    
