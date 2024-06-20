import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from optimizerSelector import defineOptimizer

class NeuralNetwork:
    def __init__(self, optimizer,learningRate):
        self.optimizer = optimizer
        self.learningRate = learningRate
    
    def createVGG16Model(self,ConvolutedLayers, denseLayers,filterLayers, maxPoolingLayers):
        # Define the VGG16 model
        model = Sequential()

        # Block 1
        model.add(Conv2D(ConvolutedLayers[0], (filterLayers[0], filterLayers[0]), activation='relu', padding='same', input_shape=(32, 32, 3)))
        model.add(Conv2D(ConvolutedLayers[1], (filterLayers[1], filterLayers[1]), activation='relu', padding='same'))
        model.add(MaxPooling2D((maxPoolingLayers[0], maxPoolingLayers[0]), strides=(2, 2)))

        # Block 2
        model.add(Conv2D(ConvolutedLayers[2], (filterLayers[2], filterLayers[2]), activation='relu', padding='same'))
        model.add(Conv2D(ConvolutedLayers[3], (filterLayers[3], filterLayers[3]), activation='relu', padding='same'))
        model.add(MaxPooling2D((maxPoolingLayers[1], maxPoolingLayers[1]), strides=(2, 2)))

        # Block 3
        model.add(Conv2D(ConvolutedLayers[4], (filterLayers[4], filterLayers[4]), activation='relu', padding='same'))
        model.add(Conv2D(ConvolutedLayers[5], (filterLayers[5], filterLayers[5]), activation='relu', padding='same'))
        model.add(Conv2D(ConvolutedLayers[6], (filterLayers[6], filterLayers[6]), activation='relu', padding='same'))
        model.add(MaxPooling2D((maxPoolingLayers[2], maxPoolingLayers[2]), strides=(2, 2)))

        # Block 4
        model.add(Conv2D(ConvolutedLayers[7], (filterLayers[7], filterLayers[7]), activation='relu', padding='same'))
        model.add(Conv2D(ConvolutedLayers[8], (filterLayers[8], filterLayers[8]), activation='relu', padding='same'))
        model.add(Conv2D(ConvolutedLayers[9], (filterLayers[9], filterLayers[9]), activation='relu', padding='same'))
        model.add(MaxPooling2D((maxPoolingLayers[3], maxPoolingLayers[3]), strides=(2, 2)))

        # Block 5
        model.add(Conv2D(ConvolutedLayers[10], (filterLayers[10], filterLayers[10]), activation='relu', padding='same'))
        model.add(Conv2D(ConvolutedLayers[11], (filterLayers[11], filterLayers[11]), activation='relu', padding='same'))
        model.add(Conv2D(ConvolutedLayers[12], (filterLayers[12], filterLayers[12]), activation='relu', padding='same'))
        model.add(MaxPooling2D((maxPoolingLayers[4], maxPoolingLayers[4]), strides=(2, 2)))

        # Flatten the output
        model.add(Flatten())

        # Fully connected layers
        model.add(Dense(denseLayers[0], activation='relu'))
        model.add(Dense(denseLayers[1], activation='relu'))
        model.add(Dense(100, activation='softmax'))

        optimizer,name = defineOptimizer(self.optimizer,self.learningRate)

        # Compile the model
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Print the model summary
        model.summary()
        
        return model
    
    def _create_mlp_model(self):
        model = Sequential()
        model.add(Dense(512, activation="relu", input_shape=(28, 28)))
        model.add(Dense(256, activation="relu"))
        model.add(Dense(10, activation="softmax"))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def _create_cnn_model(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(10, activation="softmax"))
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        return model