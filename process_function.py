import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
import random
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.backend import clear_session
import os

class AppWindow:

    def __init__(self):
        self.batch_size = 32
        self.learning_rate = 0.001
        self.optimizer = 'SGD'
        self.input_shape = (32,32,3)
        self.model = Sequential()

    
    def Load_Cifar10_dataset_Func(self):
        cifar = keras.datasets.cifar10
        (X_train, y_train), (X_test, y_test) = cifar.load_data()

        label = ['Airplane', 'Automobile', 'Bird', 'Cat',
                'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

        fig = plt.figure()
        row = 2
        column = 5
        self.input_shape = X_train[0].shape # set image shape for VGG16

        for i in range(1, row * column + 1):
            img_index = random.randint(0,50000)
            ax = fig.add_subplot(row, column, i) # add subplot to figure

            # get label index and then match the correct category in dictionary
            label_index = y_train[img_index] 
            category = label[label_index[0]]
            ax.set_title(category)

            # show random image in train data 
            img = X_train[img_index]
            plt.imshow(img)
        plt.show()

    def Load_Hyperparameter_Func(self):
        print('Hyperparameter:')
        print('batch size: ', self.batch_size)
        print('learning rate: ', self.learning_rate)
        print('optimizer: ', self.optimizer)
        print('input size: ', self.input_shape)

    def Show_Model_Structure_Func(self):
        clear_session() # use to avoid gpu outoff memory
        # conv1_1
        self.model.add(Conv2D(input_shape = self.input_shape, 
                              filters = 64, 
                              kernel_size = (3,3), 
                              padding = 'same',
                              activation = 'relu'))
        # conv1_2
        self.model.add(Conv2D(filters = 64, 
                              kernel_size = (3,3), 
                              padding = 'same',
                              activation = 'relu'))
        # max pooling
        self.model.add(MaxPooling2D(pool_size = (2,2),
                                    strides = (2,2)))
        # conv2_1
        self.model.add(Conv2D(filters = 128, 
                              kernel_size = (3,3), 
                              padding = 'same',
                              activation = 'relu'))
        # conv2_2
        self.model.add(Conv2D(filters = 128, 
                              kernel_size = (3,3), 
                              padding = 'same',
                              activation = 'relu'))
        # max pooling
        self.model.add(MaxPooling2D(pool_size = (2,2),
                                    strides = (2,2)))
        # conv3_1
        self.model.add(Conv2D(filters = 256, 
                              kernel_size = (3,3), 
                              padding = 'same',
                              activation = 'relu'))
        # conv3_2
        self.model.add(Conv2D(filters = 256, 
                              kernel_size = (3,3), 
                              padding = 'same',
                              activation = 'relu'))
        # conv3_3
        self.model.add(Conv2D(filters = 256, 
                              kernel_size = (3,3), 
                              padding = 'same',
                              activation = 'relu'))
        # max pooling
        self.model.add(MaxPooling2D(pool_size = (2,2),
                                    strides = (2,2)))
        # conv4_1
        self.model.add(Conv2D(filters = 512, 
                              kernel_size = (3,3), 
                              padding = 'same',
                              activation = 'relu'))
        # conv4_2
        self.model.add(Conv2D(filters = 512, 
                              kernel_size = (3,3), 
                              padding = 'same',
                              activation = 'relu'))
        # conv4_3
        self.model.add(Conv2D(filters = 512, 
                              kernel_size = (3,3), 
                              padding = 'same',
                              activation = 'relu'))
        # max pooling
        self.model.add(MaxPooling2D(pool_size = (2,2),
                                    strides = (2,2)))
        # conv5_1
        self.model.add(Conv2D(filters = 512, 
                              kernel_size = (3,3), 
                              padding = 'same',
                              activation = 'relu'))
        # conv5_2
        self.model.add(Conv2D(filters = 512, 
                              kernel_size = (3,3), 
                              padding = 'same',
                              activation = 'relu'))
        # conv5_3
        self.model.add(Conv2D(filters = 512, 
                              kernel_size = (3,3), 
                              padding = 'same',
                              activation = 'relu'))
        # max pooling
        self.model.add(MaxPooling2D(pool_size = (2,2),
                                    strides = (2,2)))
        # flatten the result to 4096*1 array
        self.model.add(Flatten())
        self.model.add(Dense(units = 4096,
                       activation = 'relu'))
        self.model.add(Dense(units = 4096,
                       activation = 'relu'))
        self.model.add(Dense(units = 10,
                       activation = 'softmax'))
        
        opt = SGD(self.learning_rate)
        self.model.compile(optimizer = opt, 
                           loss = keras.losses.categorical_crossentropy,
                           metrics = ['accuracy'])
        self.model.summary() # print out model structure
    
    def Show_Accuracy_and_Loss_Func(self):
        if(os.path.isfile('result.png')):
            img = cv2.imread('result.png')
            cv2.imshow('Accuracy', img)
        else:
            cifar = keras.datasets.cifar10
            (X_train, y_train), (X_test, y_test) = cifar.load_data()
            def train_y(y):
                y_one = np.zeros(10)
                y_one[y] = 1
                return y_one
            y_train_one = np.array([train_y(y_train[i]) for i in range(len(y_train))])
            y_test_one  = np.array([train_y(y_test [i]) for i in range(len(y_test ))])

            label = ['Airplane', 'Automobile', 'Bird', 'Cat',
                    'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
            
            checkpoint = ModelCheckpoint('vgg16_01.h5', monitor = 'val_accuracy',
                                        verbose = 1, save_best_only = True, 
                                        save_weights_only = False, mode = 'auto',
                                        period = 1)
            early = EarlyStopping(monitor = 'val_accuracy', min_delta = 0,
                                patience = 20, verbose = 1, 
                                mode = 'auto')
            hist = self.model.fit(X_train, y_train_one, batch_size=4, 
                                epochs=20, validation_data=(X_test, y_test_one), 
                                callbacks=[checkpoint, early])
            fig, (ax1, ax2) = plt.subplots(2, 1)

            # show accuray
            ax1.set_title('Accuracy')
            ax1.plot(hist.history['accuracy'])
            ax1.plot(hist.history['val_accuracy'])
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy')

            # show loss
            ax2.plot(hist.history['loss'])
            ax2.set_title('Loss')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')

            plt.show()
            plt.savefig('result.png')
        
    def Test_Model_Func(self, current_value):
        clear_session() # use to avoid gpu outoff memory, clear session before using
        cifar = keras.datasets.cifar10
        (X_train, y_train), (X_test, y_test) = cifar.load_data()
        def train_y(y):
            y_one = np.zeros(10)
            y_one[y] = 1
            return y_one
        y_test_one  = np.array([train_y(y_test [i]) for i in range(len(y_test ))])

        label = ['Airplane', 'Automobile', 'Bird', 'Cat',
                'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
        saved_model = load_model('vgg16_01.h5') # load pre-train model
        img = np.expand_dims(X_test[current_value], axis = 0) # because input have 4 dimension, 3 for image, 1 for label
        output = saved_model.predict(img)
        print(output[0,:])
        max_index = np.argmax(output)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        ax1.set_title(label[max_index])
        ax1.imshow(X_test[current_value])
        # ind = np.arange(5)
        ax2.bar(label, output[0,:], align = 'center')
        plt.show()
       



        

