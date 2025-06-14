from neural_network_plain_python.Dense_Layer import Dense_Layer
from neural_network_plain_python.Activation_Layer import Activation_Layer
import numpy as np
import matplotlib.pyplot as plt


class Model_plain_python:

    def __init__(self):
        self.dense1 = Dense_Layer(input_neurons=784, output_neurons=128, weights_filename="neural_network_plain_python/Weights/weights_1.txt", biases_filename="neural_network_plain_python/Biases/biases_1.txt")
        self.activation1 = Activation_Layer(type="relu")
        self.dense2 = Dense_Layer(input_neurons=128, output_neurons=64, weights_filename="neural_network_plain_python/Weights/weights_2.txt", biases_filename="neural_network_plain_python/Biases/biases_2.txt")
        self.activation2 = Activation_Layer(type="relu")
        self.dense3 = Dense_Layer(input_neurons=64, output_neurons=10, weights_filename="neural_network_plain_python/Weights/weights_3.txt", biases_filename="neural_network_plain_python/Biases/biases_3.txt")
        self.activation3 = Activation_Layer(type="softmax")


        self.layers =[
            self.dense1,
            self.activation1,
            self.dense2,
            self.activation2,
            self.dense3,
            self.activation3
        ]

        # Load weights and biases
        self.dense1.load_weights()
        self.dense1.load_biases()
        self.dense2.load_weights()
        self.dense2.load_biases()
        self.dense3.load_weights()
        self.dense3.load_biases()

    
    def predict(self, image_sample):

        input_vector = np.array(image_sample).reshape(1, 784)
         #Normalize the values
        input_vector = input_vector / 255.0
        # output = input_vector.reshape(1, -1) 
        output = input_vector.tolist()[0]

        # print(f'output length: {output}')

        for layer in self.layers:
            output = layer.forward(inputs=output)

        # get the number predicted
        predicted_num = np.argmax(output)
        print(f'full output: {output}')
        prediction_confidence = output[predicted_num]

        # Pick one sample
        # image = input_vector.reshape(28, 28)  # reshape the flat image

        # # Plot corrected sample
        # plt.imshow(image, cmap='gray')
        # plt.axhline(14, color='red')  # línea horizontal al medio
        # plt.axvline(14, color='red')  # línea vertical al medio
        # plt.title(f'True: X | Pred: {np.argmax(output)}')
        # plt.axis('off')
        # plt.show()

        return predicted_num, prediction_confidence
            
