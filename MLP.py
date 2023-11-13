
import tensorflow as tf
import random
import csv
import os
from optimizer_selector import randomize_optimizer

trains = 1000
samplingMethod = 'random'
#startingDataset = 5

for datasetNumber in range(1,21):

    datasetLocal = 'data/'+ samplingMethod +'/Fully_Connected'+ samplingMethod +'_Hyperparameters_'+ str(datasetNumber) +'.csv'
    trainingFile = 'trainings/Fully_Connected'+ samplingMethod +'/training_'+ str(datasetNumber) +'.csv'

    print('Dataset: CNN_'+ samplingMethod +'_Hyperparameters_'+ str(datasetNumber) +'.csv')
    print('Training: training_'+ str(datasetNumber) +'.csv')

    # Create a CSV file if it doesn't exist to store the results
    if not os.path.exists(trainingFile):
        with open(trainingFile, 'w', newline='') as file:
               writer = csv.writer(file)
               writer.writerow(["Optimizer","Hidden_Layer1", "Hidden_Layer2", "Learning_Rate","Batch_Size","Loss"])
               file.close()

    with open(trainingFile, 'r', newline='') as file:
        csv_reader = csv.reader(file)
        line = sum(1 for row in csv_reader)
        file.close()

    data = pd.read_csv(datasetLocal)

    L1 = data["Hidden_Layer1"].values
    L2 = data["Hidden_Layer2"].values
    optimizer_name = data["Optimizer"].values
    l_rate = data["Learning_Rate"].values
    bt_size = data["Batch_Size"].values

    for training in range(line-1, trains, 1):
          print("Training: ", training+1)
          print("Dataset: ", samplingMethod + ' ' + str(datasetNumber))
          mnist = tf.keras.datasets.mnist
          (x_train, y_train), (x_test, y_test) = mnist.load_data()

          x_train = tf.keras.utils.normalize(x_train, axis=1)
          x_test = tf.keras.utils.normalize(x_test, axis=1)

          model = tf.keras.models.Sequential()
          model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

          model.add(tf.keras.layers.Dense(L1[training], activation='ReLU'))
          model.add(tf.keras.layers.Dense(L2[training], activation='ReLU'))

          model.add(tf.keras.layers.Dense(10, activation='softmax'))

          optimizer, name = optimizer_selector.defining_optimizer_byName(optimizer_name[training], l_rate[training])

          model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
     
          epoc=5
          history = model.fit(x_train, y_train, epochs=epoc, batch_size = bt_size )

          history_dict = history.history
          print("Optmizer:", name,"Hidden Layer 1:", L1,"Hidden Layer 2:", L2,"Learning Rate:", l_rate, "Batch Size:", bt_size, "Loss:", history_dict['loss'][epoc-1])

     
          with open('data/training_FCNN_results_v2.csv', 'a', newline='') as file:
               writer = csv.writer(file)
               writer.writerow([name,L1[training], L2[training], l_rate[training],bt_size[training],history_dict['loss'][epoc-1]])
               file.close()