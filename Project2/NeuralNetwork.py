##FFNN
from typing import List
from enum import Enum
from random import random, seed
import numpy as np
import sklearn.metrics as error
from autograd import elementwise_grad
#from Scaling import ScalingAlgorithm

class ActivationFunction(Enum):
    Sigmoid = 0,
    RELU = 1,
    LeakyRELU = 2,
    Softmax = 3

#From lecture notes:
def sigmoid(X):
    try:
        return 1.0 / (1 + np.exp(-X))
    except FloatingPointError:
        return np.where(X > np.zeros(X.shape), np.ones(X.shape), np.zeros(X.shape))

def sigmoid_derivative(a):
    return a * (1 - a)
    #return elementwise_grad(sigmoid) #TODO: Look into autograd

def relu(X):
    return np.where(X > np.zeros(X.shape), X, np.zeros(X.shape))

def relu_derivative(z):
    return np.where(z > 0, 1, 0)

def leaky_relu(X):
    delta = 10e-4
    return np.where(X > np.zeros(X.shape), X, delta * X)

def leaky_relu_derivative(z):
    delta = 10e-4
    return np.where(z > 0, 1, delta)

def softmax(X):
    X = X - np.max(X, axis=-1, keepdims=True)
    delta = 10e-10
    return np.exp(X) / (np.sum(np.exp(X), axis=-1, keepdims=True) + delta)

def softmax_derivative(func):
    return elementwise_grad(func)


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

def cross_entropy(predictions, targets):
    return -(1.0 / targets.size) * np.sum(targets * np.log(predictions + 10e-10))

def cross_entropy_derivative(predictions, targets):
    epsilon = 1e-12
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    return (predictions - targets.size) / (predictions * (1 - predictions))

def mse_derivative(y_target, y_pred):
    return y_pred - y_target

def r2_derivative():
    raise NotImplementedError()

def accuracy_derivative():
    raise NotImplementedError()

class NeuralNetwork:
    def __init__(self, 
                 X_data,
                 Y_data,
                 #output_layer, 
                 cost_function, 
                 #scaler = ScalingAlgorithm.Adam,
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
            layer = HiddenLayer(hidden_activation, self.n_features, hidden_neurons, random_state, hidden_bias)
            self.hidden_layers = list([layer])
        
        n_neurons_last_hiddenlayer = self.hidden_layers[-1]._get_hidden_neurons()
        output_layer._set_weights(n_neurons_last_hiddenlayer)
        self.output_layer = output_layer


        #From ChatGPT - testing initialising ADAm
                # Adam hyperparameters
        self.alpha = 0.001   # Learning rate
        self.beta1 = 0.9     # Exponential decay rate for the first moment
        self.beta2 = 0.999   # Exponential decay rate for the second moment
        self.epsilon = 1e-8  # Small constant to prevent division by zero

        # Initialize m and v for Adam
        self.m_w = [np.zeros_like(layer.get_weights()) for layer in self.hidden_layers + [self.output_layer]]
        self.v_w = [np.zeros_like(layer.get_weights()) for layer in self.hidden_layers + [self.output_layer]]
        self.m_b = [np.zeros_like(layer.get_biases()) for layer in self.hidden_layers + [self.output_layer]]
        self.v_b = [np.zeros_like(layer.get_biases()) for layer in self.hidden_layers + [self.output_layer]]
        self.t = 1  # Time step for Adam updates


    #Function from ChaptGPT - testing
    def AdamUpdate(self, output_weights_gradient, output_bias_gradient, hidden_weights_gradient, hidden_bias_gradient):
        # Update weights and biases with Adam optimization
        layers = self.hidden_layers + [self.output_layer]
        gradients_w = hidden_weights_gradient + [output_weights_gradient]
        gradients_b = hidden_bias_gradient + [output_bias_gradient]

        for i, layer in enumerate(layers):
            # Update m and v for weights
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * gradients_w[i]
            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (gradients_w[i] ** 2)
            
            # Bias-corrected estimates
            m_w_hat = self.m_w[i] / (1 - self.beta1 ** self.t)
            v_w_hat = self.v_w[i] / (1 - self.beta2 ** self.t)
            
            # Update weights
            layer.w -= self.alpha * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
            
            # Update m and v for biases
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * gradients_b[i]
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (gradients_b[i] ** 2)
            
            # Bias-corrected estimates
            m_b_hat = self.m_b[i] / (1 - self.beta1 ** self.t)
            v_b_hat = self.v_b[i] / (1 - self.beta2 ** self.t)
            
            # Update biases
            layer.b -= self.alpha * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)
        
        # Increment time step
        self.t += 1   
    
    #So as to be flexible, we can expand the model after creation - will probably need a method like GetHiddenLayers
    def add_hidden_layer(self, layer):
        self.hidden_layers.append(layer) #TODO: At a specific position
        n_neurons_last_hiddenlayer = self.hidden_layers[-1]._get_hidden_neurons()
        self.output_layer._set_weights(n_neurons_last_hiddenlayer)
            
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

    def FeedForward(self, x):
        activations = []
        a = x  # Initial input is x
        for layer in self.hidden_layers:
            z = np.matmul(a, layer.get_weights()) + layer.get_biases()
            a = layer.activation(z)
            activations.append(a)  # Collect activation from each layer for backprop
        # Output layer computation
        z_out = np.matmul(a, self.output_layer.get_weights()) + self.output_layer.get_biases()
        a_out = self.output_layer.activation(z_out)
        activations.append(a_out)

        return activations
    
    def BackPropagation(self, activations, y, x):
        # Initializing gradients
        output_weights_gradient = []
        output_bias_gradient = []
        
        hidden_weights_gradient = []
        hidden_bias_gradient = []
        
        a_output = activations[-1]
        delta_output = self.derivative(a_output) * self.output_layer.derivative(a_output) #TODO: ?

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

        self.AdamUpdate(output_weights_gradient, output_bias_gradient, hidden_weights_gradient, hidden_bias_gradient)
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
    def train(self, epochs, learning_rate, verbosity=False):
        final_loss = 0
        #Becomes relevant if adding a stop condition:
        final_epoch = 0
        for epoch in range(epochs):
            final_epoch = epoch + 1
            # Feedforward pass
            activations = self.FeedForward(self.X_data)
            # Compute loss for monitoring (optional)
            loss = self.evaluate_model(activations[-1], self.Y_data)
            final_loss = loss
            if verbosity:
                print(f"Epoch {epoch+1}, Loss: {loss}")
            # Backpropagation
            gradients = self.BackPropagation(activations, self.Y_data, self.X_data)
            # Update weights and biases
            self.update_parameters(gradients, learning_rate)

        return [final_epoch, final_loss]
    
    def predict(self, x):
        # Perform a feedforward pass to get the final activations
        final_activation = self.FeedForward(x)[-1]
        return final_activation
    
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
    
