#& C:/Users/Lucas/anaconda3/envs/tcc/python.exe "d:/Projetos/PesquisaArtigo/TCC_Hyperparameters Optimization/CNN training.py"
import random
import csv
import os
import pandas as pd
import tensorflow as tf
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import layers
from keras import models
import optimizerSelector

trains = 20
samplingMethod = input("Whitch sampling method?")
startingDataset = int(input("Whitch sampling number?"))

datasetLocal = r'D:\Projetos\PesquisaArtigo\TCC_Hyperparameters Optimization\trainings\Fully_Connected_'+samplingMethod+'\GRPSO_'+str(startingDataset)+'particles.csv'
trainingFile = r'D:\Projetos\PesquisaArtigo\TCC_Hyperparameters Optimization\trainings\Fully_Connected_PSO\MLP_'+samplingMethod+'_'+str(startingDataset)+'_Results.csv'

# Create a CSV file if it doesn't exist to store the results
if not os.path.exists(trainingFile):
    with open(trainingFile, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Name","Hidden_Layer1", "Hidden_Layer2", "Learning_Rate","Batch_Size","Loss"])
        file.close()

with open(trainingFile, 'r', newline='') as file:
    csv_reader = csv.reader(file)
    line = sum(1 for row in csv_reader)
    file.close()

data = pd.read_csv(datasetLocal)

# Generate random hyperparameters for CNN models
L1 = data["Hidden_Layer1"].values
L2 = data["Hidden_Layer2"].values
optimizer_name = data["Name"].values
l_rate = data["Learning_Rate"].values
bt_size = data["Batch_Size"].values

# Loop over the specified number of training runs
for training in range(line-1, trains, 1):
    print("Training: ", training+1)
    #print("Dataset: ", samplingMethod + ' ' + str(datasetNumber))
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

    model.add(tf.keras.layers.Dense(L1[training], activation='ReLU'))
    model.add(tf.keras.layers.Dense(L2[training], activation='ReLU'))

    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    optimizer, name = optimizerSelector.defining_optimizer_byName(optimizer_name[training], l_rate[training])

    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
     
    epoc=5
    history = model.fit(x_train, y_train, epochs=epoc, batch_size = bt_size[training]  )

    history_dict = history.history
    print("Optmizer:", name,"Hidden Layer 1:", L1[training],"Hidden Layer 2:", L2[training],"Learning Rate:", l_rate[training], "Batch Size:", bt_size[training], "Loss:", history_dict['loss'][epoc-1])

     
    with open(trainingFile, 'a', newline='') as file:
         writer = csv.writer(file)
         writer.writerow([name,L1[training], L2[training], l_rate[training],bt_size[training],history_dict['loss'][epoc-1]])
         file.close()
