import math

class Activation_Layer:

    def __init__(self, type):
        
        if type not in ["relu", "softmax"]:
            raise ValueError("Activation layer type must be either 'relu' or 'softmax'")    

        self.type = type

        if type == "relu":
            self.activation = self.relu
            self.activation_prime = self.relu_prime
        elif type == "softmax":
            self.activation = self.softmax
            self.activation_prime = None

        return

    def forward(self, inputs):
        self.inputs = inputs
        output = []
        for x in inputs:
            val = self.activation(x=x)
            output.append(val)

        return output
    

    def backward(self, gradients, learning_rate):
        if self.type == "softmax":
            return gradients
        dE_dX = []
        for grad, x in zip(gradients, self.inputs):
            local_dE_dx = grad * self.activation_prime(x)
            dE_dX.append(local_dE_dx) 

        return dE_dX
    


    def relu(self, x):
        return max(0, x)
    
    def relu_prime(self, x):
        if x > 0: return 1
        else: return 0
    

    def softmax(self, x):
        max_val = max(self.inputs)
        exp_sum = 0
        for val in self.inputs:
            exp_sum += math.exp(val-max_val)
        output = math.exp(x-max_val) / exp_sum
        return output