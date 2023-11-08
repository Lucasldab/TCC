import random
import csv
import os
import pandas as pd
import tensorflow as tf
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import layers
from keras import models
import optimizer_selector


#start = 1#int(input("Which row it starts: "))
trains = 1000#int(input("CNN numbers: "))
datasetNumber = 6
samplingMethod = 'LHS'
datasetLocal = 'data/'+ samplingMethod +'/CNN_'+ samplingMethod +'_Hyperparameters_'+ str(datasetNumber) +'.csv'
trainingFile = 'trainings/CNN_'+ samplingMethod +'/training_'+ str(datasetNumber) +'.csv'

#print(start)
#print(trains)

# Create a CSV file if it doesn't exist to store the results
if not os.path.exists(trainingFile):
    with open(trainingFile, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Convoluted_Layers1", "Convoluted_Filters1", "Convoluted_Layers2", "Convoluted_Filters2", "Hidden_Layer1", "Hidden_Layer2", "Learning_Rate", "Batch_Size", "Loss"])
        file.close()

with open(trainingFile, 'r', newline='') as file:
    csv_reader = csv.reader(file)
    start = sum(1 for row in csv_reader)

data = pd.read_csv(datasetLocal)

# Generate random hyperparameters for CNN models
conv_n1 = data["Convoluted_Layers1"].values
conv_f1 = data["Convoluted_Filters1"].values
conv_n2 = data["Convoluted_Layers2"].values
conv_f2 = data["Convoluted_Filters2"].values
L1 = data["Hidden_Layer1"].values
L2 = data["Hidden_Layer2"].values
optimizer_name = data["Name"].values
l_rate = data["Learning_Rate"].values
bt_size = data["Batch_Size"].values

# Loop over the specified number of training runs
for training in range(start-1, trains, 1):
    print("Training:", training+1)

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
    optimizer, name = optimizer_selector.defining_optimizer_byName(optimizer_name[training], l_rate[training])
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    epoc = 5

    # Train the model
    history = model.fit(x_train, y_train, epochs=epoc, batch_size=bt_size[training])

    # Extract the loss from the training history
    history_dict = history.history
    final_loss = history_dict['loss'][epoc-1]

    print("Dataset: ", samplingMethod + str(datasetNumber))

    # Print and save model and training information
    print("Optimizer: {} Convoluted Layers 1: {} Convoluted Filters 1: {} Convoluted Layers 2: {} Convoluted Filters 2: {}\n Hidden Layer 1: {} Hidden Layer 2: {} Learning Rate: {} Batch Size: {} Loss: {}".format(name, conv_n1[training], conv_f1[training], conv_n2[training], conv_f2[training], L1[training], L2[training], l_rate[training], bt_size[training], final_loss))
    
    with open(trainingFile, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, conv_n1[training], conv_f1[training], conv_n2[training], conv_f2[training], L1[training], L2[training], l_rate[training], bt_size[training], final_loss])
        file.close()