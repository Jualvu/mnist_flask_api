import numpy as np

class Layer_Dense:

    def __init__(self, n_input_neurons, n_output_neurons):
        self.input_neurons_size = n_input_neurons
        self.output_neurons_size = n_output_neurons 
        #generate a matrix of weights with random small values
        self.weights = 0.1 * np.random.randn(n_output_neurons, n_input_neurons)  
        #generate a vector of biases 
        self.biases = np.random.randn(n_output_neurons, 1) 

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(self.inputs, self.weights.T) + self.biases.T # execute the matrix multiplication
        # notice that we didnt have to transpose weights because of the size choice in the class construction
        return self.output
    
    def backward(self, output_gradient, learning_rate):

        """
            Steps:

            For gradient Descent we need to calculate: dE_dW and dE_db

            Then, we need to calculate dE_dX to pass it to previous layer

            Remember:

            output_gradient = dE_dY

            dE_dW = dE_dY * X.T
            dE_db = dE_dY

            dE_dX = W.T * dE_dY
        
        """
        dE_dW = np.dot(output_gradient.T, self.inputs) # dE_dY * X.T
        dE_db = output_gradient.T
        #update parameters w and b
        self.weights -= learning_rate * dE_dW
        self.biases -= learning_rate * dE_db

        #calculate dE_dX
        dE_dX = np.dot(output_gradient, self.weights)
        return dE_dX
    
    def load_weights(self, file_name):
        weights = np.load(file_name)
        self.weights = weights

    def load_biases(self, file_name):
        biases = np.load(file_name)
        self.biases = biases

            