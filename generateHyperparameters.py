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
trains = 50000
#trains2 = 50000
dataSamples = 20
for samples in range(0,dataSamples):

    csvName = 'data/sobol/CNN_Sobol_Hyperparameters_'+ str(samples+1) +'.csv'

    # Create a CSV file if it doesn't exist to store the results
    if not os.path.exists(csvName):
        with open(csvName, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Name", "Convoluted_Layers1", "Convoluted_Filters1", "Convoluted_Layers2", "Convoluted_Filters2", "Hidden_Layer1", "Hidden_Layer2", "Learning_Rate", "Batch_Size"])
            file.close()

    # Generate random hyperparameters for CNN models
    conv_n1 = Sampling.sobol_sampling(trains, 16., 32.)
    conv_f1 = Sampling.sobol_sampling(trains, 1., 3.)
    conv_n2 = Sampling.sobol_sampling(trains, 16., 32.)
    conv_f2 = Sampling.sobol_sampling(trains, 1., 3.)
    L1 = Sampling.sobol_sampling(trains, 32., 64.)
    L2 = Sampling.sobol_sampling(trains, 16., 32.)
    optimizer_number = Sampling.sobol_sampling(trains, 1., 9.)
    l_rate = Sampling.sobol_sampling(trains, 0.0001, 0.01, round=False)
    bt_size = Sampling.sobol_sampling(trains, 32., 128.)

    #print(len(optimizer_number))
    #break
    for training in range(0, trains2, 1):
        print("Training:", training + 1)
        optimizer, name = randomize_optimizer(optimizer_number[training], l_rate[training])
    
        with open(csvName, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([name, conv_n1[training], conv_f1[training], conv_n2[training], conv_f2[training], L1[training], L2[training], l_rate[training], bt_size[training]])
            file.close()