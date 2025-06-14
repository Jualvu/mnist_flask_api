import numpy as np
import Layer_Activation

class ReLU:

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



