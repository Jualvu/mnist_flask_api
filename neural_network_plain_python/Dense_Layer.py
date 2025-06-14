import math
import random

class Dense_Layer:

    def __init__(self, input_neurons, output_neurons, weights_filename, biases_filename):
        self.weights_filename = weights_filename
        self.biases_filename = biases_filename
        self.input_neurons = input_neurons
        self.output_neurons = output_neurons
        self.weights_filename = weights_filename
        self.biases_filename = biases_filename

        def he_init(n_inputs):
            return random.gauss(0, math.sqrt(2 / n_inputs))


        #Initialize weights and biases
        weights = []
        biases = []

        for i in range(output_neurons):
            local_w = []
            for j in range(input_neurons):
                w = he_init(n_inputs=input_neurons)
                local_w.append(w)
            local_b = random.uniform(-0.01, 0.01)
            weights.append(local_w)
            biases.append(local_b)

        self.weights = weights
        self.biases = biases



    def forward(self, inputs):
        assert len(inputs) == self.input_neurons, "inputs size has to match expected samples"

        self.inputs = inputs

        output = []
        for i in range(len(self.weights)):
            neuron_output = 0
            for j in range(len(self.weights[i])):
                neuron_output += self.weights[i][j] * inputs[j]
            neuron_output += self.biases[i]
            output.append(neuron_output)

        return output
    

    def backward(self, gradients, learning_rate):
        # 2D matrix
        dE_dw = []
        for i in range(len(gradients)):
            local_dE_dw = []
            for j in range(len(self.inputs)):
                val = self.inputs[j] * gradients[i]
                local_dE_dw.append(val)
            dE_dw.append(local_dE_dw)

        #same as gradients
        dE_db = gradients
        
        #learning
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                self.weights[i][j] -= learning_rate * dE_dw[i][j]

        for i in range(len(self.biases)):
            self.biases[i] -= learning_rate * dE_db[i]

        # 1D array of same size as number of neurons (dot product)
        # multiply weights by column
        dE_dX = []

        for i in range(len(self.weights[0])):
            local_dE_dx = 0
            for j in range(len(self.weights)):
                local_dE_dx += self.weights[j][i] * gradients[j]
            dE_dX.append(local_dE_dx)


        return dE_dX
    



    def save_weights(self):
        with open(self.weights_filename, 'w') as f:
            for i in range(len(self.weights)):
                weights_str = ','.join(str(w) for w in self.weights[i])
                f.write(weights_str + '\n')

    def save_biases(self):
        with open(self.biases_filename, 'w') as f:
            for i in range(len(self.biases)):
                f.write(str(self.biases[i]) + '\n')



    def load_weights(self):
        with open(self.weights_filename, 'r') as f:
            lines = f.readlines() 
            for i in range(len(self.weights)):
                weights = [float(w) for w in lines[i].strip().split(',')]
                self.weights[i] = weights

    def load_biases(self):
        with open(self.biases_filename, 'r') as f:
            lines = f.readlines() 
            for i in range(len(self.biases)):
                bias = float(lines[i].strip())
                self.biases[i] = bias