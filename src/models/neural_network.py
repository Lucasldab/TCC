"""
Neural network module implementing various architectures.
"""
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from src.optimizers.optimizer_selector import define_optimizer

class NeuralNetwork:
    """Base class for neural network architectures."""
    
    def __init__(self, optimizer: str, learning_rate: float):
        """
        Initialize the neural network with optimizer and learning rate.
        
        Args:
            optimizer: Name of the optimizer to use
            learning_rate: Learning rate for the optimizer
        """
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.optimizer = define_optimizer(self.optimizer, self.learning_rate)
    
    def create_vgg16_model(self, conv_layers: list, dense_layers: list, 
                          filter_layers: list, max_pooling_layers: list) -> Sequential:
        """
        Create a VGG16-style convolutional neural network.
        
        Args:
            conv_layers: List of number of filters for each convolutional layer
            dense_layers: List of number of units for each dense layer
            filter_layers: List of filter sizes for each convolutional layer
            max_pooling_layers: List of pool sizes for each max pooling layer
            
        Returns:
            Compiled Keras Sequential model
        """
        model = Sequential()

        # Block 1
        model.add(Conv2D(conv_layers[0], (filter_layers[0], filter_layers[0]), 
                        activation='relu', padding='same', input_shape=(32, 32, 3)))
        model.add(Conv2D(conv_layers[1], (filter_layers[1], filter_layers[1]), 
                        activation='relu', padding='same'))
        model.add(MaxPooling2D((max_pooling_layers[0], max_pooling_layers[0]), 
                             strides=(2, 2)))

        # Block 2
        model.add(Conv2D(conv_layers[2], (filter_layers[2], filter_layers[2]), 
                        activation='relu', padding='same'))
        model.add(Conv2D(conv_layers[3], (filter_layers[3], filter_layers[3]), 
                        activation='relu', padding='same'))
        model.add(MaxPooling2D((max_pooling_layers[1], max_pooling_layers[1]), 
                             strides=(2, 2), padding='same'))

        # Block 3
        model.add(Conv2D(conv_layers[4], (filter_layers[4], filter_layers[4]), 
                        activation='relu', padding='same'))
        model.add(Conv2D(conv_layers[5], (filter_layers[5], filter_layers[5]), 
                        activation='relu', padding='same'))
        model.add(Conv2D(conv_layers[6], (filter_layers[6], filter_layers[6]), 
                        activation='relu', padding='same'))
        model.add(MaxPooling2D((max_pooling_layers[2], max_pooling_layers[2]), 
                             strides=(2, 2), padding='same'))

        # Block 4
        model.add(Conv2D(conv_layers[7], (filter_layers[7], filter_layers[7]), 
                        activation='relu', padding='same'))
        model.add(Conv2D(conv_layers[8], (filter_layers[8], filter_layers[8]), 
                        activation='relu', padding='same'))
        model.add(Conv2D(conv_layers[9], (filter_layers[9], filter_layers[9]), 
                        activation='relu', padding='same'))
        model.add(MaxPooling2D((max_pooling_layers[3], max_pooling_layers[3]), 
                             strides=(2, 2), padding='same'))

        # Block 5
        model.add(Conv2D(conv_layers[10], (filter_layers[10], filter_layers[10]), 
                        activation='relu', padding='same'))
        model.add(Conv2D(conv_layers[11], (filter_layers[11], filter_layers[11]), 
                        activation='relu', padding='same'))
        model.add(Conv2D(conv_layers[12], (filter_layers[12], filter_layers[12]), 
                        activation='relu', padding='same'))
        model.add(MaxPooling2D((max_pooling_layers[4], max_pooling_layers[4]), 
                             strides=(2, 2), padding='same'))

        # Flatten the output
        model.add(Flatten())

        # Fully connected layers
        model.add(Dense(dense_layers[0], activation='relu'))
        model.add(Dense(dense_layers[1], activation='relu'))
        model.add(Dense(100, activation='softmax'))

        # Compile the model
        model.compile(optimizer=self.optimizer[0], 
                     loss='sparse_categorical_crossentropy', 
                     metrics=['accuracy'])

        return model
    
    def create_mlp_model(self, dense_layers: list) -> Sequential:
        """
        Create a Multi-Layer Perceptron model.
        
        Args:
            dense_layers: List of number of units for each dense layer
            
        Returns:
            Compiled Keras Sequential model
        """
        model = Sequential()
        model.add(Dense(dense_layers[0], activation="relu", input_shape=(28, 28)))
        model.add(Dense(dense_layers[1], activation="relu"))
        model.add(Dense(10, activation="softmax"))
        model.compile(optimizer=self.optimizer, 
                     loss='sparse_categorical_crossentropy', 
                     metrics=['accuracy'])
        return model
    
    def create_cnn_model(self, conv_layers: list, dense_layers: list, 
                        filter_layers: list, max_pooling_layers: list = [2, 2]) -> Sequential:
        """
        Create a Convolutional Neural Network model.
        
        Args:
            conv_layers: List of number of filters for each convolutional layer
            dense_layers: List of number of units for each dense layer
            filter_layers: List of filter sizes for each convolutional layer
            max_pooling_layers: List of pool sizes for each max pooling layer
            
        Returns:
            Compiled Keras Sequential model
        """
        model = Sequential()
        model.add(Conv2D(conv_layers[0], 
                        kernel_size=(filter_layers[0], filter_layers[0]), 
                        activation="relu", 
                        input_shape=(28, 28, 1)))
        model.add(MaxPooling2D(pool_size=(max_pooling_layers[0], max_pooling_layers[0])))
        model.add(Conv2D(conv_layers[1], 
                        kernel_size=(filter_layers[1], filter_layers[1]), 
                        activation="relu"))
        model.add(MaxPooling2D(pool_size=(max_pooling_layers[1], max_pooling_layers[1])))
        model.add(Flatten())
        model.add(Dense(dense_layers[0], activation="relu"))
        model.add(Dense(dense_layers[1], activation="relu"))
        model.add(Dense(10, activation="softmax"))
        model.compile(optimizer=self.optimizer, 
                     loss='categorical_crossentropy', 
                     metrics=['accuracy'])
        return model 