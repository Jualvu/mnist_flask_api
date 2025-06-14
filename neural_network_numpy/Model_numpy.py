from neural_network_numpy.Layer_Activation import Layer_Activation
from neural_network_numpy.Layer_Dense import Layer_Dense 
import numpy as np
import matplotlib.pyplot as plt

class Model_numpy:

    def __init__(self):
        self.dense1 = Layer_Dense(n_input_neurons=784, n_output_neurons=128)
        self.activation1 = Layer_Activation(type='relu')
        self.dense2 = Layer_Dense(n_input_neurons=128, n_output_neurons=64)
        self.activation2 = Layer_Activation(type='relu')
        self.dense3 = Layer_Dense(n_input_neurons=64, n_output_neurons=10)
        self.activation3 = Layer_Activation(type='softmax') # activation prime is None for softmax because we will use crossentropyLoss

        self.layers= [
            self.dense1,
            self.activation1,
            self.dense2,
            self.activation2,
            self.dense3,
            self.activation3
        ]

        #LOAD WEIGHTS AND BIASES
        self.dense1.load_weights("neural_network_numpy/values/w_layer1.npy")
        self.dense1.load_biases("neural_network_numpy/values/b_layer1.npy")
        self.dense2.load_weights("neural_network_numpy/values/w_layer2.npy")
        self.dense2.load_biases("neural_network_numpy/values/b_layer2.npy")
        self.dense3.load_weights("neural_network_numpy/values/w_layer3.npy")
        self.dense3.load_biases("neural_network_numpy/values/b_layer3.npy")


    def predict(self, image_sample):
        input_vector = np.array(image_sample).reshape(1, 784)

        print(f"input shape: {input_vector.shape}")
        print(f'vector before normalizing: {input_vector}')

        #Normalize the values
        input_vector = input_vector / 255.0

        # np.save('neural_network/values/sampletest.npy', input_vector)
        print(f'input vector after normalizing: \n{input_vector}')


        # output = input_vector.reshape(1, -1) 
        output = input_vector

        print(f'output (input vector reshaped): \n{output}')

        


        for layer in self.layers:
            #keep propagating forward the output values from layer to layer
            output = layer.forward(inputs=output)


        # Pick one sample
        # image = input_vector.reshape(28, 28)  # reshape the flat image

        # # Plot corrected sample
        # plt.imshow(image, cmap='gray')
        # plt.axhline(14, color='red')  # línea horizontal al medio
        # plt.axvline(14, color='red')  # línea vertical al medio
        # plt.title(f'True: X | Pred: {np.argmax(output)}')
        # plt.axis('off')
        # plt.show()


        print(f'final output shape: {output.shape}')

        # get the number predicted
        predicted_num = np.argmax(output)
        prediction_confidence = output[0][predicted_num]

        return predicted_num, prediction_confidence
        




