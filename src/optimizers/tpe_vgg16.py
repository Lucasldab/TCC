import os
import csv
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.optimizers import SGD, RMSprop, Adam, AdamW, Adadelta, Adagrad, Adamax, Nadam, Ftrl
import tensorflow.keras.backend as backend
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import neuralNetwork  # Assuming neuralNetwork.py is in the same directory

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

def objective_function_VGG16(params):
    # Extract hyperparameters from the 'params' dictionary
    name = params['Name']
    ConvolutedLayers = [int(params['Convoluted_Layers{}'.format(i+1)]) for i in range(13)]
    denseLayers = [int(params['Dense_Layer{}'.format(i+1)]) for i in range(2)]
    filterLayers = [int(params['Convoluted_Filters{}'.format(i+1)]) for i in range(13)]
    maxPoolingLayers = [int(params['Max_Pooling_Layers{}'.format(i+1)]) for i in range(5)]
    learning_rate = params['Learning_Rate']
    batch_size = int(params['Batch_Size'])
    epoch = 5
    NN = neuralNetwork.NeuralNetwork(optimizer=name,learningRate=learning_rate)
    model = NN.createVGG16Model(ConvolutedLayers=ConvolutedLayers,denseLayers=denseLayers,filterLayers=filterLayers,maxPoolingLayers=maxPoolingLayers)

    # Train the model
    history = model.fit(x_train, y_train, epochs=epoch, batch_size=batch_size, verbose=1)

    # Extract the loss from the training history
    final_loss = history.history['loss'][-1]
    backend.clear_session()

    return {'loss': final_loss, 'status': STATUS_OK}
    
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

spaceVGG16 = {
    'Name': hp.choice('Name', ['SGD', 'RMSprop', 'Adam', 'AdamW', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam', 'Ftrl']),
    'Convoluted_Layers1': hp.quniform('Convoluted_Layers1', 32, 128, 1),
    'Convoluted_Layers2': hp.quniform('Convoluted_Layers2', 32, 128, 1),
    'Convoluted_Layers3': hp.quniform('Convoluted_Layers3', 64, 256, 1),
    'Convoluted_Layers4': hp.quniform('Convoluted_Layers4', 64, 256, 1),
    'Convoluted_Layers5': hp.quniform('Convoluted_Layers5', 128, 512, 1),
    'Convoluted_Layers6': hp.quniform('Convoluted_Layers6', 128, 512, 1),
    'Convoluted_Layers7': hp.quniform('Convoluted_Layers7', 128, 512, 1),
    'Convoluted_Layers8': hp.quniform('Convoluted_Layers8', 256, 1024, 1),
    'Convoluted_Layers9': hp.quniform('Convoluted_Layers9', 256, 1024, 1),
    'Convoluted_Layers10': hp.quniform('Convoluted_Layers10', 256, 1024, 1),
    'Convoluted_Layers11': hp.quniform('Convoluted_Layers11', 512, 2048, 1),
    'Convoluted_Layers12': hp.quniform('Convoluted_Layers12', 512, 2048, 1),
    'Convoluted_Layers13': hp.quniform('Convoluted_Layers13', 512, 2048, 1),
    'Convoluted_Filters1': hp.quniform('Convoluted_Filters1', 1, 3, 1),
    'Convoluted_Filters2': hp.quniform('Convoluted_Filters2', 1, 3, 1),
    'Convoluted_Filters3': hp.quniform('Convoluted_Filters3', 1, 3, 1),
    'Convoluted_Filters4': hp.quniform('Convoluted_Filters4', 1, 3, 1),
    'Convoluted_Filters5': hp.quniform('Convoluted_Filters5', 1, 3, 1),
    'Convoluted_Filters6': hp.quniform('Convoluted_Filters6', 1, 3, 1),
    'Convoluted_Filters7': hp.quniform('Convoluted_Filters7', 1, 3, 1),
    'Convoluted_Filters8': hp.quniform('Convoluted_Filters8', 1, 3, 1),
    'Convoluted_Filters9': hp.quniform('Convoluted_Filters9', 1, 3, 1),
    'Convoluted_Filters10': hp.quniform('Convoluted_Filters10', 1, 3, 1),
    'Convoluted_Filters11': hp.quniform('Convoluted_Filters11', 1, 3, 1),
    'Convoluted_Filters12': hp.quniform('Convoluted_Filters12', 1, 3, 1),
    'Convoluted_Filters13': hp.quniform('Convoluted_Filters13', 1, 3, 1),
    'Max_Pooling_Layers1': hp.quniform('Max_Pooling_Layers1', 1, 3, 1),
    'Max_Pooling_Layers2': hp.quniform('Max_Pooling_Layers2', 1, 3, 1),
    'Max_Pooling_Layers3': hp.quniform('Max_Pooling_Layers3', 1, 3, 1),
    'Max_Pooling_Layers4': hp.quniform('Max_Pooling_Layers4', 1, 3, 1),
    'Max_Pooling_Layers5': hp.quniform('Max_Pooling_Layers5', 1, 3, 1),
    'Dense_Layer1': hp.quniform('Dense_Layer1', 32, 64, 1),
    'Dense_Layer2': hp.quniform('Dense_Layer2', 16, 32, 1),
    'Learning_Rate': hp.loguniform('Learning_Rate', np.log(0.0001), np.log(0.01)),
    'Batch_Size': hp.quniform('Batch_Size', 32, 128, 1),
}


trains = 100
samplingMethod = 'TPE'
model = 'VGG16'
totalOfDataset = 20

for datasetNumber in range(totalOfDataset):
    folderName = 'trainings/' + model + '_' + samplingMethod
    trainingFile = 'trainings/' + model + '_' + samplingMethod + '/training_' + str(datasetNumber+1) + '.csv'

    print('Training: training_' + str(datasetNumber) + '.csv')

    # Create a CSV file if it doesn't exist to store the results
    if not os.path.exists(trainingFile):
        os.makedirs(folderName)
        with open(trainingFile, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Optimizer", "Learning_Rate", "Batch_Size", "Convoluted_Layers1", "Convoluted_Layers2", "Convoluted_Layers3", "Convoluted_Layers4", "Convoluted_Layers5", "Convoluted_Layers6", "Convoluted_Layers7", "Convoluted_Layers8", "Convoluted_Layers9", "Convoluted_Layers10", "Convoluted_Layers11", "Convoluted_Layers12", "Convoluted_Layers13", "Dense_Layers1", "Dense_Layers2", "Filter_Layers1", "Filter_Layers2", "Filter_Layers3", "Filter_Layers4", "Filter_Layers5", "Filter_Layers6", "Filter_Layers7", "Filter_Layers8", "Filter_Layers9", "Filter_Layers10", "Filter_Layers11", "Filter_Layers12", "Filter_Layers13", "Max_Pooling_Layers1", "Max_Pooling_Layers2", "Max_Pooling_Layers3", "Max_Pooling_Layers4", "Max_Pooling_Layers5", "Loss"])
            file.close()

    with open(trainingFile, 'r', newline='') as file:
        csv_reader = csv.reader(file)
        line = sum(1 for row in csv_reader)
        file.close()

    for training in range(line - 1, trains, 1):
        # Create a Trials object to keep track of the optimization process
        trials = Trials()

        # Use TPE to search for the best hyperparameters
        best = fmin(fn=objective_function_VGG16,
                    space=spaceVGG16,
                    algo=tpe.suggest,
                    max_evals=5,  # Adjust the number of evaluations as needed
                    trials=trials)

        print("Best hyperparameters:", best)

        # Extract best parameters and loss
        losses = [trial['result']['loss'] for trial in trials.trials]
        best_trial_index = np.argmin(losses)
        best_params = trials.trials[best_trial_index]['misc']['vals']
        best_params = {key[0]: val[0] for key, val in best_params.items()}

        # Append the best hyperparameters and loss to the CSV file
        with open(trainingFile, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([best_params['Name'], best_params['Learning_Rate'], best_params['Batch_Size'],
                             best_params['Convoluted_Layers1'], best_params['Convoluted_Layers2'], best_params['Convoluted_Layers3'],
                             best_params['Convoluted_Layers4'], best_params['Convoluted_Layers5'], best_params['Convoluted_Layers6'],
                             best_params['Convoluted_Layers7'], best_params['Convoluted_Layers8'], best_params['Convoluted_Layers9'],
                             best_params['Convoluted_Layers10'], best_params['Convoluted_Layers11'], best_params['Convoluted_Layers12'],
                             best_params['Convoluted_Layers13'], best_params['Dense_Layer1'], best_params['Dense_Layer2'],
                             best_params['Convoluted_Filters1'], best_params['Convoluted_Filters2'], best_params['Convoluted_Filters3'],
                             best_params['Convoluted_Filters4'], best_params['Convoluted_Filters5'], best_params['Convoluted_Filters6'],
                             best_params['Convoluted_Filters7'], best_params['Convoluted_Filters8'], best_params['Convoluted_Filters9'],
                             best_params['Convoluted_Filters10'], best_params['Convoluted_Filters11'], best_params['Convoluted_Filters12'],
                             best_params['Convoluted_Filters13'], best_params['Max_Pooling_Layers1'], best_params['Max_Pooling_Layers2'],
                             best_params['Max_Pooling_Layers3'], best_params['Max_Pooling_Layers4'], best_params['Max_Pooling_Layers5'],
                             losses[best_trial_index]])
            file.close()