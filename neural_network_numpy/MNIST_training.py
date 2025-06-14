import numpy as np
import pandas as pd
import os
print(os.getcwd())
from Layer_Dense import Layer_Dense 
from CategoricalCrossentropyLoss import CategoricalCrossentropyLoss 
from Layer_Activation import Layer_Activation

#import mnist dataset
mnist_dataset = pd.read_csv('mnist_dataset/mnist_train.csv')

#separate features and labels
X = mnist_dataset.iloc[:, 1:]  # dont use first column since its y value
y = mnist_dataset.iloc[:, 0:1] # use only first column since its y value



#print X and y
print("X dataset shape: " + str(X.shape))
print("y dataset shape: " + str(y.shape))
print("\n")

def save_weights(file_name, weights):
    np.save(file_name, weights)

def save_biases(file_name, biases):
    np.save(file_name, biases)

#Create the model
"""
    Neural Network architecture

    2 hidden layers
    1 hidden layer of 128 neurons
    2 hidden layer of 64 neurons
    output 10 neurons for softmax

"""

dense1 = Layer_Dense(n_input_neurons=784, n_output_neurons=128)
activation1 = Layer_Activation(type='relu')
dense2 = Layer_Dense(n_input_neurons=dense1.output_neurons_size, n_output_neurons=64)
activation2 = Layer_Activation(type='relu')
dense3 = Layer_Dense(n_input_neurons=dense2.output_neurons_size, n_output_neurons=10)
activation3 = Layer_Activation(type='softmax') # activation prime is None for softmax because we will use crossentropyLoss


layers= [
    dense1,
    activation1,
    dense2,
    activation2,
    dense3,
    activation3
]

print("layers shapes:")
for layer in layers:
    if(type(layer) == Layer_Dense):
        print("(" + str(layer.input_neurons_size) + ", "+ str(layer.output_neurons_size) + ")")



#Normalize the X values
X = X / 255.0

print("x.shape")
print(X.shape)
print("y.shape")
print(y.shape)


#LOAD WEIGHTS AND BIASES
dense1.load_weights("values/w_layer1.npy")
dense1.load_biases("values/b_layer1.npy")
dense2.load_weights("values/w_layer2.npy")
dense2.load_biases("values/b_layer2.npy")
dense3.load_weights("values/w_layer3.npy")
dense3.load_biases("values/b_layer3.npy")


epochs = 1
learning_rate = 0.0001

X = X.to_numpy()
y = y.to_numpy()


#train the model
print("\nTRAINING\n")
for i in range(epochs):

    # First do forward propagation
    # 1. To get prediction results
    # 2. To calculate loss
    # 3. To start backprop
    error = 0
    correct_predictions = 0
    # print(len(X))

    #calculate predictions on each m example
    for x, y_true in zip(X, y):
        # FORWARD PROP
        #initiliaze output values with the m sample from the x tranining data
        output = x.reshape(1, -1) 
        # print("output shape")
        # print(output.shape)
        # output = x

        for layer in layers:
            #keep propagating forward the output values from layer to layer
            output = layer.forward(inputs=output)

        # print("output:")
        # print(output.shape)
        # print(output)
        #one hot encoding y_true
        y_true_one_hot_encoded = np.zeros(output.shape)
        y_true_one_hot_encoded[0, y_true] = 1.0
        
        # print("y_true_one_hot_encoded:")
        # print(y_true_one_hot_encoded.shape)
        # print(y_true_one_hot_encoded)

        #with the output we can calculate CategoricalCrossentropyLoss to then get dE_dY
        loss = CategoricalCrossentropyLoss.forward(y_prediction=output, y_true=y_true_one_hot_encoded)
        # if not np.isfinite(loss):
        #     print("Loss is not finite:", loss)
        #     break

        error += loss


        # print("loss: ")
        # print(loss)

        if np.argmax(output) == y_true:
            correct_predictions += 1

        # print("\nerror:")
        # print(error)
        # print("\nPredicted num: ")
        # print(np.argmax(output))
        # print("\nActual num: ")
        # print(y)

        # print("\nfull output:\n")
        # i = 0
        # for num in output[0]:
        #     print("i: " + str(i) + "  ---> " + str(num))
        #     i += 1

        # print("sum:")
        # print(np.sum(output[0]))




        # BACK PROP

        #start backpropagation by calculating dE_dY with output


        dE_dY = output - y_true_one_hot_encoded  # softmax + cross-entropy simplification
        # print("dE_dY.shape")
        # print(dE_dY.shape)

        # i = 0
        for layer in reversed(layers):
            # print("layer : " + str(i))
            # i += 1
            dE_dY = layer.backward(output_gradient=dE_dY, learning_rate=learning_rate)

    # Print every 10 epochs (you can change the interval)
    # if i % 2 == 0:
    print("error")
    print(str(error) + "\n")
    average_loss = error / len(X)
    accuracy = correct_predictions / len(X)
    print(f"Epoch {i}: Loss = {average_loss:.4f}, Accuracy = {accuracy:.4f}")



# Code to save weights and biases after training
print(f"Saving weights and biases:\n")

j = 3
for layer in reversed(layers):
        if type(layer) == Layer_Dense:
            # save weights
            save_weights(file_name=f"values/w_layer{j}.npy", weights=layer.weights)
            save_biases(file_name=f"values/b_layer{j}.npy", biases=layer.biases)
            j -= 1





