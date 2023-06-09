import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dropout, BatchNormalization, Flatten, Dense, MaxPool2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.layers import Resizing
from tensorflow.keras.applications import MobileNetV2

class AffectNetModel:
    def __init__(self, variant = 0):
        self.INPUT_PATH = "data/affectNet/"
        self.data_dir = self.INPUT_PATH
        self.IMAGE_SIZE = (96, 96)
        self.model = None
        self.variant = variant
        self.color_dim = 3 if self.variant >= 2 else 1
        #self.color_dim = 1
        self.fix_gpu()

    def fix_gpu(self):
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.set_visible_devices(physical_devices[0], 'GPU')

    def load_data(self):
        images = []
        labels = []
        label_map = {0: 'surprise', 1:'fear', 2:'neutral', 3:'sad', 4:'disgust', 5:'contempt', 6:'happy', 7:'anger'}

        for label in label_map:
            category_path = os.path.join(self.data_dir, label_map[label])  # Updated line
            for img_name in os.listdir(category_path):
                img_path = os.path.join(category_path, img_name)
                if(self.variant < 2):
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                else:
                    img = cv2.imread(img_path)
                img = cv2.resize(img, self.IMAGE_SIZE)
                images.append(img)
                labels.append(label)

        images = np.array(images, dtype="float32") / 255.0
        images = images.reshape(images.shape[0], *self.IMAGE_SIZE, self.color_dim)  # Add color dimension (1 for grayscale images)
        labels = np.array(labels, dtype="int32")
        labels = to_categorical(labels, num_classes=len(label_map))
        return images, labels


    #def prepare_data(self):
    #    images, labels = self.load_data()
    #    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)
    #    return X_train, X_val, y_train, y_val

    def prepare_data(self):
        images, labels = self.load_data()
        X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
        return X_train, X_val, X_test, y_train, y_val, y_test


    def plot_history(self, history):
        pd.DataFrame(history.history).plot()
        plt.title("Training history")
        plt.xlabel("Epochs")
        plt.ylabel("Metrics")
        plt.legend()
        plt.show()

    def create_model_v1(self, input_shape):
        model = Sequential()

        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.1))

        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.1))

        model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.1))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.2))

        model.add(Dense(8, activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print(model.summary())
        model.summary()
        return model
    
    def create_model_v2(self, input_shape):
        model = Sequential()
        model.add(Conv2D(32, (3,3), activation="relu", input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3,3), activation="relu"))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3,3), activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, (3,3), activation="relu"))
        model.add(BatchNormalization())
        model.add(Conv2D(128, (3,3), activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(256, (3,3), activation="relu"))
        model.add(BatchNormalization())
        model.add(Conv2D(256, (3,3), activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(8, activation='softmax'))

        model.compile(optimizer = Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

        model.summary()
        return model

    def create_model_v4(self, input_shape):
        model = Sequential()
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(MaxPool2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(MaxPool2D(pool_size=(2, 2)))

        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(MaxPool2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(8, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
        model.summary()
        return model
    

    def create_model_v3(self, input_shape):
        """
            ResNet model
        """
        pretrained_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, classes=8, pooling='avg', input_shape=input_shape)
        pretrained_model.trainable = False

        model = Sequential()
        model.add(pretrained_model)
        
        print(model.layers)
        print(input_shape)

        model.add(Conv2D(32, (3,3), activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Dropout(0.25))
        
        model.add(Flatten())
        model.add(Dense(8, activation='softmax'))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.00010609141674890813)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        return model

    def train(self):
        #X_train, X_val, y_train, y_val = self.prepare_data()
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data()
        input_shape = (*self.IMAGE_SIZE, self.color_dim)
        #input_shape = (*self.IMAGE_SIZE, 3)

        if(self.variant == 1):
            self.model = self.create_model_v2(input_shape)
        elif(self.variant == 2):
            self.model = self.create_model_v3(input_shape)
        elif(self.variant == 3):
            self.model = self.create_model_v4(input_shape)
        #elif(self.variant == 3):
        #    self.model = self.create_model_v4(input_shape) # scratch model
        else: # else is mainly for variant == 0
            self.model = self.create_model_v1(input_shape)

        history = self.model.fit(X_train, y_train,
                                    epochs=30,
                                    batch_size=32,
                                    validation_data=(X_val, y_val),
                                    callbacks=[EarlyStopping(patience=10, monitor='val_loss', mode='min'), 
                                            ReduceLROnPlateau(patience=2, verbose=1),
                                            ModelCheckpoint('best_model.h5', 
                                                            save_best_only=True, 
                                                            save_weights_only=True, 
                                                            monitor='val_accuracy', 
                                                            mode='max')],
                                    verbose=1)


        test_loss, test_accuracy = self.model.evaluate(X_test, y_test)
        print("Test Loss:", test_loss)
        print("Test Accuracy:", test_accuracy)


        self.plot_history(history)
        
        if(self.variant == 10):
            self.model.save('affectNet_v2.h5')
        elif(self.variant == 2):
            self.model.save('affectNet_v3.h5')
        elif(self.variant == 3):
            self.model.save('affectNet_v4.h5')
        #elif(self.variant == 3):
        #    self.model.save('affectNet_v4.h5')
        else:
            self.model.save('affectNet_v1.h5')

if __name__ == "__main__":
    affect_net_model = AffectNetModel(variant = 3)
    affect_net_model.train()
