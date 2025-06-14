import numpy as np

class Layer_Activation:

    def __init__(self, type: str):

        if(type == 'relu'):
            self.activation = self.relu
            self.activation_prime = self.relu_prime
        elif(type == 'softmax'):
            self.activation = self.softmax
            self.activation_prime = None
        else:
            print("Error: Activation function not provided")

    def forward(self, inputs):
        # just apply activation function
        self.inputs = inputs
        return self.activation(self.inputs)


    def backward(self, output_gradient, learning_rate): 
        if self.activation_prime is None:
            return output_gradient
        # we have learning rate parameter to standarize every layer's backward method parameters
        #here we need to calculate dE_dX = dE_dY * activation_prime(x)
        # print("output_gradient shape from activation:")
        # print(output_gradient.shape)
        dE_dX = output_gradient * self.activation_prime(self.inputs)
        # print("dE_dX shape from activation:")
        # print(dE_dX.shape)
        return dE_dX
    

    # ReLU functions
    def relu(self, inputs):
        return np.maximum(0, inputs)

    def relu_prime(self, inputs):
        #relu derivative
        return (inputs > 0).astype(float) 
        """
            This implementation goes over every input value in inputs
            and ask the boolean if statement (inputs[i] > 0) 
            if true return 1
            else return 0
        """



    #Softmax functions
    def softmax(self, inputs):
        #first exponential every value
        #additional, subtract max value from current m input example to each input value in that m example
        exponential_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        #second find the sum of each m example
        sums = np.sum(exponential_values, axis=1, keepdims=True)
        """
            Quick note on np.sum

            axis=1 means that the sum is going to be done on each row
            axis=0 means that the sum is going to be done on each column
            keepdims(keep dimensions)=True means that the shape is going to persist
            so, if its summing up each row, its going to be 3 rows  [
                                                                    [2],
                                                                    [3],
                                                                    [4]
                                                                    ]
        """
        probabilities = exponential_values / sums # apply formula
        return probabilities
