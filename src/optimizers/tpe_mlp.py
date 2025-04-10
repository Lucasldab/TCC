import pandas as pd
import tensorflow as tf
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import layers
from keras import models
from keras import optimizers
import optimizerSelector
import csv
import dataTreatment
import os
import numpy as np
import hyperopt
import pandas as pd
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import neuralNetwork

def objective_function_MLP(params):
    # Extract hyperparameters from the 'params' dictionary
    name = params['Name']
    hidden_layer1 = int(params['Hidden_Layer1'])
    hidden_layer2 = int(params['Hidden_Layer2'])
    learning_rate = float(params['Learning_Rate'])
    batch_size = int(params['Batch_Size'])
    
    epoc = 5
    denseLayers = [hidden_layer1,hidden_layer2]
    NN = neuralNetwork.NeuralNetwork(optimizer=name,learningRate=learning_rate)
    model = NN.createMLPmodel(denseLayers=denseLayers)

    # Train the model
    history = model.fit(x_train, y_train, epochs=epoc, batch_size=batch_size)

    # Extract the loss from the training history
    history_dict = history.history
    final_loss = history_dict['loss'][epoc-1]
    
    
    return {'loss': final_loss, 'status': hyperopt.STATUS_OK}
    
# Load and preprocess MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape((60000, 28, 28, 1))
x_train = x_train.astype('float32') / 255
x_test = x_test.reshape((10000, 28, 28, 1))
x_test = x_test.astype('float32') / 255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

spaceMLP = {
    'Name': hp.choice('Name', ['SGD', 'RMSprop', 'Adam', 'AdamW', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam', 'Ftrl']),  # Categorical hyperparameter
    'Hidden_Layer1': hp.quniform('Hidden_Layer1', 32, 64, 1),
    'Hidden_Layer2': hp.quniform('Hidden_Layer2', 16, 32, 1),
    'Learning_Rate': hp.loguniform('Learning_Rate', 0.0001, 0.01),
    'Batch_Size': hp.quniform('Batch_Size', 32, 128, 1),
}

trains = 100
samplingMethod = 'TPE'
totalOfDataset = 20

for datasetNumber in totalOfDataset:

    trainingFile = 'trainings/Fully_Connected_'+ samplingMethod +'/training_'+ str(datasetNumber) +'.csv'
    
    print('Dataset: CNN_'+ samplingMethod +'_Hyperparameters_'+ str(datasetNumber) +'.csv')
    print('Training: training_'+ str(datasetNumber) +'.csv')
    
    # Create a CSV file if it doesn't exist to store the results
    if not os.path.exists(trainingFile):
        with open(trainingFile, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Name", "Hidden_Layer1", "Hidden_Layer2", "Learning_Rate", "Batch_Size", "Loss"])
            file.close()
    with open(trainingFile, 'r', newline='') as file:
        csv_reader = csv.reader(file)
        line = sum(1 for row in csv_reader)
        file.close()
    
    for training in range(line-1, trains,1):
        print("Training: ", training+1)
        # Create a Trials object to keep track of the optimization process
        trials = Trials()
    
        # Use TPE to search for the best hyperparameters
        best = fmin(fn=objective_function_MLP,
                    space=spaceMLP,
                    algo=tpe.suggest,
                    max_evals=5,  # Adjust the number of evaluations as needed
                    trials=trials)
    
        print("Best hyperparameters:", best)
    
        best = pd.DataFrame.from_dict(best, orient='index')
        best = best.transpose().reset_index()
        best = best.drop('index', axis=1)
    
        losses = [trial['result']['loss'] for trial in trials.trials]
        final_loss = np.min(losses)
        print('Training: training_'+ str(datasetNumber) +'.csv')
        print("Name: {} Hidden_Layer1: {} Hidden_Layer2: {} Learning_Rate: {} Batch_Size: {} Loss: {}".format(optimizerSelector.numberToName(int(best['Name'][0])),int(best['Hidden_Layer1'][0]),int(best['Hidden_Layer2'][0]),best['Learning_Rate'][0],int(best['Batch_Size'][0]),final_loss))
    
        with open(trainingFile, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([optimizerSelector.numberToName(int(best['Name'][0])),int(best['Hidden_Layer1'][0]),int(best['Hidden_Layer2'][0]),best['Learning_Rate'][0],int(best['Batch_Size'][0]),final_loss])
            file.close()