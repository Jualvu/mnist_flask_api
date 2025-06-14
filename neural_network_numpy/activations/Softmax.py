import numpy as np

class Softmax:

    def softmax(self, inputs):
        self.inputs = inputs

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

    def softmax_prime():
        return