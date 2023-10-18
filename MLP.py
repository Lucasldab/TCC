
import tensorflow as tf
import random
import csv
import os
from optimizer_selector import randomize_optimizer

if not os.path.exists('data/training_FCNN_results_v2.csv'):
     with open('data/training_FCNN_results_v2.csv', 'w', newline='') as file:
          writer = csv.writer(file)
          writer.writerow(["Optimizer","Hidden_Layer1", "Hidden_Layer2", "Learning_Rate","Batch_Size","Loss"])
          file.close()

for training in range(999, 1000, 1):
     print("Training: ",training)
     mnist = tf.keras.datasets.mnist
     (x_train, y_train), (x_test, y_test) = mnist.load_data()

     x_train = tf.keras.utils.normalize(x_train, axis=1)
     x_test = tf.keras.utils.normalize(x_test, axis=1)

     model = tf.keras.models.Sequential()
     model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

     L1 = random.randint(32, 256)
     model.add(tf.keras.layers.Dense(L1, activation='ReLU'))

     L2 = random.randint(32, 256)
     model.add(tf.keras.layers.Dense(L2, activation='ReLU'))

     model.add(tf.keras.layers.Dense(10, activation='softmax'))

     optimizer, l_rate, name = randomize_optimizer()

     model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
 
     bt_size = random.randint(32, 128)
     epoc=5
     history = model.fit(x_train, y_train, epochs=epoc, batch_size = bt_size )

     history_dict = history.history
     print("Optmizer:", name,"Hidden Layer 1:", L1,"Hidden Layer 2:", L2,"Learning Rate:", l_rate, "Batch Size:", bt_size, "Loss:", history_dict['loss'][epoc-1])

     
     with open('data/training_FCNN_results_v2.csv', 'a', newline='') as file:
          writer = csv.writer(file)
          writer.writerow([name,L1, L2, l_rate,bt_size,history_dict['loss'][epoc-1]])
    
     file.close()