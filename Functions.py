import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

class DataHandler:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def PlotDataWithFittedCurve(self, x_values, y_values, title):
        plt.plot(x_values, y_values, 'b-', label='Fitted curve')
        plt.plot(self.x, self.y ,'ro', label='Data')
        plt.axis([0,1.0,0, self.y.max()])
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')
        plt.title(title)
        plt.legend()
        plt.show()

    
    def Predict(self, xModel):
        #From ChatGPT:
        # Create the design matrix X
        X = np.column_stack((self.x**2, self.x, np.ones_like(self.x)))

        # Compute the coefficients using the normal equation
        beta = np.linalg.inv(X.T @ X) @ (X.T @ self.y)

        # The coefficients a, b, c
        a, b, c = beta

        y_predict = a*xModel**2 + b*xModel + c

        return y_predict
        
    def SkLearnPredict(self, polynomial_order):
        #Parametrise using scikit-learn
        poly = PolynomialFeatures(degree = polynomial_order, include_bias=False)
        poly_model = LinearRegression()

        poly_features = poly.fit_transform(self.x.reshape(-1, 1))
        poly_model.fit(poly_features, self.y)

        x_model = np.linspace(0, 1.0, 100).reshape(-1, 1)
        x_model_poly = poly.transform(x_model)

        yPredict = poly_model.predict(x_model_poly)

        return (x_model, yPredict)

    def SkLearnFeatureMatrix(self, polynomial_order):
        poly = PolynomialFeatures(degree = polynomial_order, include_bias=False)
        poly_model = LinearRegression()

        poly_features = poly.fit_transform(self.x) #.reshape(-1, 1))
        poly_model.fit(poly_features, self.y)

        x_model = np.linspace(0, 1.0, 100) #.reshape(-1, 1)

        return poly_features