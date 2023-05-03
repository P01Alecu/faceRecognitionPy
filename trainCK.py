import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os,cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pylab import rcParams
rcParams['figure.figsize'] = 20, 10

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import keras

from keras.utils import np_utils
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Activation , Dropout ,Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.models import model_from_json
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import BatchNormalization
import os
from keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import tensorflow as tf

class AffectNetModel:
    def __init__(self):
        self.INPUT_PATH = "data/ckPlus/"
        self.data_dir = self.INPUT_PATH
        self.IMAGE_SIZE = (48, 48)
        self.model = None
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
        label_map = {0: 'anger', 1:'contempt', 2:'disgust', 3:'fear', 4:'happy', 5:'sadness', 6:'surprise'}

        for label in label_map:
            category_path = os.path.join(self.data_dir, label_map[label])  # Updated line
            for img_name in os.listdir(category_path):
                img_path = os.path.join(category_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, self.IMAGE_SIZE)
                images.append(img)
                labels.append(label)

        images = np.array(images, dtype="float32") / 255.0
        images = images.reshape(images.shape[0], *self.IMAGE_SIZE, 1)  # Add color dimension (1 for grayscale images)
        labels = np.array(labels, dtype="int32")
        labels = to_categorical(labels, num_classes=len(label_map))
        return images, labels


    def prepare_data(self):
        images, labels = self.load_data()
        X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)
        return X_train, X_val, y_train, y_val

    def plot_history(self, history):
        pd.DataFrame(history.history).plot()
        plt.title("Training history")
        plt.xlabel("Epochs")
        plt.ylabel("Metrics")
        plt.legend()
        plt.show()

    def create_model(self, input_shape):

        model = Sequential()
        model.add(Conv2D(6, (5, 5), input_shape=input_shape, padding='same', activation = 'relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(16, (5, 5), padding='same', activation = 'relu'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3), activation = 'relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(128, activation = 'relu'))
        model.add(Dropout(0.5))
        model.add(Dense(7, activation = 'softmax'))

        model.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer='adam')

        model.summary()
        return model

    def train(self):
        X_train, X_val, y_train, y_val = self.prepare_data()
        input_shape = (*self.IMAGE_SIZE, 1)

        self.model = self.create_model(input_shape)

        history = self.model.fit(X_train, y_train,
                                    epochs=100,
                                    validation_data=(X_val, y_val),
                                    callbacks=[EarlyStopping(patience=10, monitor='val_loss', mode='min'), 
                                            ReduceLROnPlateau(patience=2, verbose=1),
                                            ModelCheckpoint('best_model.h5', 
                                                            save_best_only=True, 
                                                            save_weights_only=True, 
                                                            monitor='val_accuracy', 
                                                            mode='max')],
                                    verbose=1)

        self.plot_history(history)
        
        self.model.save('ckPlus.h5')

if __name__ == "__main__":
    affect_net_model = AffectNetModel()
    affect_net_model.train()
