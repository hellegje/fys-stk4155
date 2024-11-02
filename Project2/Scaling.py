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
            if beta_values is None:
                raise ValueError("Beta values must be provided to run with ADAM")
            
            #From ChatGPT:
            t += 1  # Increment time step
            first_moment = beta_values[0] * first_moment + (1 - beta_values[0]) * gradients  # Update first moment
            second_moment = beta_values[1] * second_moment + (1 - beta_values[1]) * gradients**2  # Update second moment

            # Bias correction
            first_term = first_moment / (1 - beta_values[0]**t)  # Corrected first moment
            second_term = second_moment / (1 - beta_values[1]**t)  # Corrected second moment

            update = eta * first_term / (np.sqrt(second_term) + delta)  # Update parameters

            beta -= update
            
        case _:
            raise NotImplementedError()

