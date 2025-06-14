from Dense_Layer import Dense_Layer
from Activation_Layer import Activation_Layer
from CategoricalCrossentropyLoss import CategoricalCrossentropyLoss
import pandas as pd
import numpy as np
import math

#import mnist dataset
mnist_dataset = pd.read_csv('mnist_dataset/mnist_test.csv')

#separate features and labels
# trained on batches
X = mnist_dataset.iloc[10000:12000, 1:]
y = mnist_dataset.iloc[10000:12000, 0:1]


X = X / 255

X_normalized = X.values.tolist()
y = y.values.tolist()




dense1 = Dense_Layer(input_neurons=784, output_neurons=128, weights_filename="Weights/weights_1.txt", biases_filename="Biases/biases_1.txt")
activation1 = Activation_Layer(type="relu")
dense2 = Dense_Layer(input_neurons=128, output_neurons=64, weights_filename="Weights/weights_2.txt", biases_filename="Biases/biases_2.txt")
activation2 = Activation_Layer(type="relu")
dense3 = Dense_Layer(input_neurons=64, output_neurons=10, weights_filename="Weights/weights_3.txt", biases_filename="Biases/biases_3.txt")
activation3 = Activation_Layer(type="softmax")


layers =[
    dense1,
    activation1,
    dense2,
    activation2,
    dense3,
    activation3
]

# Load weights and biases
dense1.load_weights()
dense1.load_biases()
dense2.load_weights()
dense2.load_biases()
dense3.load_weights()
dense3.load_biases()

#Training Loop

epochs = 10



for i in range(epochs):

    error = 0
    correct_predictions = 0

    for x_sample, y_sample in zip(X_normalized, y):

        output = x_sample
        #FORWARD PASS
        for layer in layers:
            output = layer.forward(output)

        
        # one-hot
        y_true = [0] * 10
        y_true[y_sample[0]] = 1

        true_class_index = y_true.index(1)
        predicted_class_index = output.index(max(output))

        #check correct predictions
        if true_class_index == predicted_class_index:
            correct_predictions += 1

        # calcular loss con softmax ya aplicado
        loss = CategoricalCrossentropyLoss.calculate(y_pred=output,y_true=y_true)
        error += loss


    print(f"\nTotal ERROR: {error}")

    average_loss = error / len(X_normalized)
    accuracy = correct_predictions / len(X_normalized)
    print(f"Epoch {i}: Avarage Loss = {average_loss:.4f}, Accuracy = {accuracy * 100:.2f}%\n")








