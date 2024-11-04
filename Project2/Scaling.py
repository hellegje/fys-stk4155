import numpy as np
from enum import Enum


class ScalingAlgorithm(Enum):
    GradientDescent = 0,
    MomentumGD = 1, #TODO: This is probably not relevant here anymore
    StochasticGD = 2,
    Adagrad = 3,
    RMSprop = 4,
    Adam = 5

def scale(scaling_algorithm, momentum, gradients, eta):
    delta = 1e-8
    match scaling_algorithm:
        case ScalingAlgorithm.StochasticGD:
            if momentum:
                change = momentum*change - eta*gradients
                beta += change
            else:
                beta -= eta*gradients

        case ScalingAlgorithm.Adagrad:
            G_iter += gradients * gradients
            update = gradients*eta/(delta+np.sqrt(G_iter))
            if momentum:
                change = momentum*change - update
                beta += change
            else:
                beta -= update

        case ScalingAlgorithm.RMSprop:
            if rho is None:
                raise ValueError("Decay rate cannot be null when scaling with RMSprop")
            G_iter = rho*G_iter + (1-rho) * gradients**2
            update = gradients*eta/(delta+np.sqrt(G_iter))
            if momentum:
                change = momentum*change - update
                beta += change
            else:
                beta -= update
        
        case ScalingAlgorithm.Adam:
        case _:
            raise NotImplementedError()

#From ChatGPT
class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # First moment vector
        self.v = {}  # Second moment vector
        self.t = 0   # Time step for bias correction

    def initialize(self, params):
        # Initialize moment vectors for each parameter
        for key, param in params.items():
            self.m[key] = np.zeros_like(param)
            self.v[key] = np.zeros_like(param)

    def update(self, params, grads):
        self.t += 1
        updated_params = {}
        for key in params:
            # Update biased first moment estimate
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            # Update biased second moment estimate
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)
            # Compute bias-corrected first moment estimate
            m_corrected = self.m[key] / (1 - self.beta1 ** self.t)
            # Compute bias-corrected second moment estimate
            v_corrected = self.v[key] / (1 - self.beta2 ** self.t)
            # Update parameters
            updated_params[key] = params[key] - self.lr * m_corrected / (np.sqrt(v_corrected) + self.epsilon)
        return updated_params

            

