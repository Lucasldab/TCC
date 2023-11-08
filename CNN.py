import random
import csv
import os
import tensorflow as tf
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import layers
from keras import models
from optimizer_selector import randomize_optimizer
import Sampling

# Define the number of CNN models to train
trains = int(input("CNN numbers: "))

# Create a CSV file if it doesn't exist to store the results
if not os.path.exists('trainings/CNN_LHS_9Hyper.csv'):
    with open('trainings/CNN_LHS_9Hyper.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Convoluted_Layers1", "Convoluted_Filters1", "Convoluted_Layers2", "Convoluted_Filters2", "Hidden_Layer1", "Hidden_Layer2", "Learning_Rate", "Batch_Size", "Loss"])
        file.close()

# Generate random hyperparameters for CNN models
conv_n1 = Sampling.lhs_sampling(trains, 16., 32.)
conv_f1 = Sampling.lhs_sampling(trains, 1., 3.)
conv_n2 = Sampling.lhs_sampling(trains, 16., 32.)
conv_f2 = Sampling.lhs_sampling(trains, 1., 3.)
L1 = Sampling.lhs_sampling(trains, 32., 64.)
L2 = Sampling.lhs_sampling(trains, 16., 32.)
optimizer_number = Sampling.lhs_sampling(trains, 1., 9.)
l_rate = Sampling.lhs_sampling(trains, 0.0001, 0.01, round=False)
bt_size = Sampling.lhs_sampling(trains, 32., 128.)

# Loop over the specified number of training runs
for training in range(0, trains, 1):
    print("Training:", training + 1)

    # Load and preprocess MNIST data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape((60000, 28, 28, 1))
    x_train = x_train.astype('float32') / 255
    x_test = x_test.reshape((10000, 28, 28, 1))
    x_test = x_test.astype('float32') / 255
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # Create a Sequential model
    model = models.Sequential()

    # Add convolutional layers
    model.add(layers.Conv2D(int(conv_n1[training]), (int(conv_f1[training])), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(int(conv_n2[training]), (int(conv_f2[training])), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())

    # Add fully connected layers
    model.add(layers.Dense(L1[training], activation='relu'))
    model.add(layers.Dense(L2[training], activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    # Compile the model with a random optimizer
    optimizer, name = randomize_optimizer(optimizer_number[training], l_rate[training])
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    epoc = 5

    # Train the model
    history = model.fit(x_train, y_train, epochs=epoc, batch_size=bt_size[training])

    # Extract the loss from the training history
    history_dict = history.history
    final_loss = history_dict['loss'][epoc-1]

    # Print and save model and training information
    print("Optimizer: {} Convoluted Layers 1: {} Convoluted Filters 1: {} Convoluted Layers 2: {} Convoluted Filters 2: {} Hidden Layer 1: {} Hidden Layer 2: {} Learning Rate: {} Batch Size: {} Loss: {}".format(name, conv_n1[training], conv_f1[training], conv_n2[training], conv_f2[training], L1[training], L2[training], l_rate[training], bt_size[training], final_loss))
    
    with open('trainings/CNN_LHS_9Hyper.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, conv_n1[training], conv_f1[training], conv_n2[training], conv_f2[training], L1[training], L2[training], l_rate[training], bt_size[training], final_loss])
        file.close()