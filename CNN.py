
import random
import csv
import os
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import layers
from keras import models
from keras import losses
from optimizer_selector import randomize_optimizer


if not os.path.exists('data/training_CNN_results_v3.csv'):
     with open('data/training_CNN_results_v3.csv', 'w', newline='') as file:
          writer = csv.writer(file)
          writer.writerow(["Name","Convoluted_Layers1","Convoluted_Filters1","Convoluted_Layers2","Convoluted_Filters2","Hidden_Layer1", "Hidden_Layer2", "Learning_Rate","Batch_Size","Loss"])
          file.close()

for training in range(0,1000,1):
    print("Training: ",training)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape((60000,28,28,1))
    x_train = x_train.astype('float32')/255

    x_test = x_test.reshape((10000,28,28,1))
    x_test = x_test.astype('float32')/255

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    model = models.Sequential()

    conv_n1 = random.randint(16, 32)
    conv_f1 = random.randint(1, 3)
    model.add(layers.Conv2D(conv_n1,(conv_f1), activation='relu', input_shape = (28,28,1)))
    model.add(layers.MaxPooling2D((2,2)))
    conv_n2 = random.randint(16, 32)
    conv_f2 = random.randint(1, 3)
    model.add(layers.Conv2D(conv_n2,(conv_f2), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Flatten())
    L1 = random.randint(32, 64)
    model.add(layers.Dense(L1,activation = 'relu'))
    L2 = random.randint(16, 32)
    model.add(layers.Dense(L2,activation = 'relu'))
    model.add(layers.Dense(10, activation= 'softmax'))

    #model.summary()

    optimizer, l_rate, name = randomize_optimizer()

    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    bt_size = random.randint(32, 128)
    epoc=5
    history = model.fit(x_train, y_train, epochs=epoc, batch_size = bt_size )

    history_dict = history.history
    print("Optmizer:{} Convoluted Layers 1:{} Convoluted Filters 1:{} Convoluted Layers 2:{} Convoluted Filters 2:{}  Hidden Layer 1:{} Hidden Layer 2:{} Learning Rate:{} Batch Size:{} Loss:{}".format(name,conv_n1,conv_f1,conv_n2, conv_f2,L1,L2,l_rate,bt_size,history_dict['loss'][epoc-1]))

    with open('data/training_CNN_results_v3.csv', 'a', newline='') as file:
         writer = csv.writer(file)
         writer.writerow([name,conv_n1,conv_f1,conv_n2, conv_f2,L1,L2,l_rate,bt_size,history_dict['loss'][epoc-1]])
         file.close()