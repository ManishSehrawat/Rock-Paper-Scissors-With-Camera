# NO NOT MODIFY THE CONTENTS OF THIS FILE UNLESS REQUIRED

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

CLASS_MAP = {
    "empty": 0,
    "rock": 1,
    "paper": 2,
    "scissors": 3
}

def mapper(value):
    return CLASS_MAP[value]

def create_model(input_shape, num_classes):
    model = keras.models.Sequential([

        keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=input_shape),
        keras.layers.MaxPooling2D(2, 2),
        # The second convolution
        keras.layers.Conv2D(64, (3,3), activation='relu'),
        keras.layers.MaxPooling2D(2,2),
        # The third convolution
        keras.layers.Conv2D(128, (3,3), activation='relu'),
        keras.layers.MaxPooling2D(2,2),
        # The fourth convolution
        keras.layers.Conv2D(128, (3,3), activation='relu'),
        keras.layers.MaxPooling2D(2,2),
        # The fifth convolution
        keras.layers.Conv2D(128, (3,3), activation='relu'),
        keras.layers.MaxPooling2D(2,2),
        # The sixth convolution
        keras.layers.Conv2D(256, (3,3), activation='relu'),
        keras.layers.MaxPooling2D(2,2),
        # Flatten the results to feed into a DNN
        keras.layers.Flatten(),
        keras.layers.Dropout(0.5),
        # 512 neuron hidden layer
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

def generate_model(X, y):

    dataset, labels = X, y

    # label encode the classes
    labels = list(map(mapper, labels))

    input_shape = (225, 225, 3)
    model = create_model(input_shape, len(CLASS_MAP))

    with tf.device('/device:CPU:0'):

        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=keras.optimizers.Adam(1e-3, amsgrad=True),
                      metrics=['accuracy'])

        model.summary()

        # start training
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.5, random_state=42)
        
        history = model.fit(np.array(X_train),np.array(y_train),validation_data=(np.array(X_test),np.array(y_test)),epochs=9)
        
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(acc) + 1)
        
        # Train and validation accuracy
        plt.plot(epochs, acc, 'b', label='Training accurarcy')
        plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
        plt.title('Training and Validation accurarcy')
        plt.legend()
        
        plt.figure()
        
        # Train and validation loss
        plt.plot(epochs, loss, 'b', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and Validation loss')
        plt.legend()
        plt.show()

        # save the model for using it later to detect the action performed by user on the camera
        model.save("rock-paper-scissors-model.h5")
len(CLASS_MAP)