##FFNN
from typing import List
from enum import Enum
from random import random, seed
import numpy as np
import sklearn.metrics as error

class ActivationFunction(Enum):
    Sigmoid = 0,
    RELU = 1,
    LeakyRELU = 2,
    Softmax = 3

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

def softmax(z):
    exp_values = np.exp(z - np.max(z, axis=1, keepdims=True))
    probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    return probabilities

class CostFunction(Enum):
    MSE = 0,
    R2 = 1,
    Cross_Entropy = 2

def mse(y_target, y_pred):
    n = np.size(y_pred)  # Number of data points
    return np.sum((y_target - y_pred)**2)/n

def r2(y_target, y_pred):
    return error.r2_score(y_target, y_pred)

def accuracy(y_target, y_pred):
    return error.accuracy_score(y_target, y_pred)

def mse_derivative(y_target, y_pred):
    return y_pred - y_target

def r2_derivative():
    raise NotImplementedError()

def accuracy_derivative():
    raise NotImplementedError()

def cross_entropy(predictions, targets):
    epsilon = 1e-12
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    return -np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))

def cross_entropy_derivative(predictions, targets):
    epsilon = 1e-12
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    return (predictions - targets) / (predictions * (1 - predictions))

class NeuralNetwork:
    def __init__(self, 
                 X_data,
                 Y_data,
                 #output_layer, 
                 cost_function, 
                 random_state = None,
                 #Output layer:
                 output_activation = None,
                 n_outputs = 1, 
                 output_bias = 0,
                 #Hidden layer:
                 hidden_activation = ActivationFunction.Sigmoid,
                 hidden_neurons = 1,
                 hidden_bias = 0,
                 hidden_layers = None):
        

        self.n_inputs = X_data.shape[0]
        self.n_features = X_data.shape[1] if len(X_data.shape) > 1 else 1
        self.n_hidden_neurons = hidden_neurons
        #self.n_categories = n_categories

        self.X_data = X_data
        self.Y_data = Y_data
        #self.input_layer = input_layer
        self.hidden_layers = hidden_layers
        self.cost_function = cost_function
        output_layer = OutputLayer(output_activation, n_outputs, random_state, output_bias)

        #User can self-define a list of hidden layers, otherwise the default is one layer with given values
        if hidden_layers == None:
            layer = HiddenLayer(hidden_activation, hidden_neurons, random_state, hidden_bias)
            self.hidden_layers = list([layer])
        
        n_neurons_last_hiddenlayer = self.hidden_layers[-1]._get_hidden_neurons()
        output_layer._set_weights(n_neurons_last_hiddenlayer)
        self.output_layer = output_layer
            
    #TODO: Consider moving the weights to the initialisation of the model rather than the layer
    #So as to be flexible, we can expand the model after creation - will probably need a method like GetHiddenLayers
    def add_hidden_layer(self, position, layer):
        self.hidden_layers.append(layer) #TODO: At a specific position
            
    def evaluate_model(self, predicted_value, target_value):
        match self.cost_function:
            case CostFunction.MSE:
                return mse(target_value, predicted_value)
            case CostFunction.R2:
                return r2(target_value, predicted_value)
            case CostFunction.Cross_Entropy:
                return cross_entropy(target_value, predicted_value)
            case _:
                raise NotImplementedError(f"Cost function {self.cost_function} not implemented.")

    #For training
    def feed_forward(self):
        activations = []
        a = self.X_data  # Initial input is x
        for layer in self.hidden_layers:
            z = np.matmul(a, layer.get_weights()) + layer.get_biases()
            a = layer.activation(z)
            activations.append(a)  # Collect activation from each layer for backprop
        # Output layer computation
        z_out = np.matmul(a, self.output_layer.get_weights()) + self.output_layer.get_biases()
        a_out = self.output_layer.activation(z_out)
        activations.append(a_out)
        return activations
    
    #For output
    def feed_forward_out(self, X):
        a = X
        for layer in self.hidden_layers:
            z = np.matmul(a, layer.get_weights()) + layer.get_biases()
            a = layer.activation(z)
        # Output layer computation
        z_out = np.matmul(a, self.output_layer.get_weights()) + self.output_layer.get_biases()
        a_out = self.output_layer.activation(z_out)

        return a_out
    
    def BackPropagation(self, activations):
        # Initializing gradients
        output_weights_gradient = []
        output_bias_gradient = []
        
        hidden_weights_gradient = []
        hidden_bias_gradient = []
        
        a_output = activations[-1]
        delta_output = self.derivative(a_output) #* self.output_layer.derivative(a_output)


        # Backpropagate through hidden layers
        delta = np.matmul(delta_output, self.output_layer.get_weights().T) * self.hidden_layers[-1].derivative(activations[-2])
        output_weights_gradient = np.matmul(activations[-2].T, delta_output)
        output_bias_gradient = np.sum(delta_output, axis=0, keepdims=True)

        for i in reversed(range(len(self.hidden_layers))):
            # Insert at start of list because working backwards
            hidden_weights_gradient.insert(0, np.matmul(activations[i-1].T, delta) if i > 0 else np.matmul(self.X_data.T, delta))
            hidden_bias_gradient.insert(0, np.sum(delta, axis=0, keepdims=True))

            # Prepare delta for the next layer back
            if i > 0:
                delta = np.matmul(delta, self.hidden_layers[i].get_weights().T) * self.hidden_layers[i-1].derivative(activations[i-1])
        
        return output_weights_gradient, output_bias_gradient, hidden_weights_gradient, hidden_bias_gradient

    def derivative(self, a):
        y = self.Y_data
        match self.cost_function:
            case CostFunction.MSE:
                return mse_derivative(y, a)
            case CostFunction.R2:
                return r2_derivative(y, a)
            case CostFunction.Cross_Entropy:
                return cross_entropy_derivative(y, a)
            case _:
                raise NotImplementedError(f"Derivative function not implemented for {self.activation_fnc}")
    
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
    def train(self, epochs, learning_rate, verbosity=False):
        final_loss = 0
        #Becomes relevant if adding a stop condition:
        final_epoch = 0
        for epoch in range(epochs):
            final_epoch = epoch + 1
            # Feedforward pass
            activations = self.feed_forward()
            # Compute loss for monitoring (optional)
            loss = self.evaluate_model(activations[-1], self.Y_data)
            final_loss = loss
            if verbosity:
                print(f"Epoch {epoch+1}, Loss: {loss}")
            # Backpropagation
            gradients = self.BackPropagation(activations)
            # Check gradients for NaNs
            if any(np.isnan(g).any() for g in gradients):
                print("NaN detected in gradients during backpropagation.")
                break
            # Update weights and biases
            self.update_parameters(gradients, learning_rate)

        return [final_epoch, final_loss]
    
    def predict(self, X):
        # Perform a feedforward pass to get the final activations
        return self.feed_forward_out(X)

    
class Layer:
    def __init__(self, activation_function, n_features, n_hidden_neurons, random_state=None, hidden_bias=0):
        # weights and bias in the hidden layer
        if random_state is not None and random_state >= 0:
            np.random.seed(random_state)
        self.w = np.random.randn(n_features, n_hidden_neurons)
        self.b = np.zeros((1, n_hidden_neurons)) + hidden_bias
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
            case ActivationFunction.Softmax:
                return softmax(z)
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
    def __init__(self, activation_function, n_outputs, random_state=None, output_bias=0):
        # weights and bias in the output layer
        self.random_state = random_state
        self.n_outputs = n_outputs
        self.b = np.zeros((1, n_outputs)) + output_bias
        self.activation_fnc = activation_function

    def _set_weights(self, n_hidden_neurons):
        if self.random_state is not None and self.random_state >= 0:
            np.random.seed(self.random_state)
        self.w = np.random.randn(n_hidden_neurons, self.n_outputs)

#TODO: Input layer if/when needed
    
