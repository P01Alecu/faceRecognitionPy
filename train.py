import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.layers import Resizing
import tensorflow_hub as hub

class EmotionClassifier:
    def __init__(self, train_data_dir, validation_data_dir, url=''):
        self.train_data_dir = train_data_dir
        self.validation_data_dir = validation_data_dir
        self.IMG_HEIGHT = 48
        self.IMG_WIDTH = 48
        self.batch_size = 32
        self.class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        self.fix_gpu()
        self.train_generator, self.validation_generator = self.prepare_data()
        self.url = url
        if(len(self.url) > 3):
            self.model = self.build_model_from_url()
        else:
            self.model = self.build_model()
        
    def fix_gpu(self):
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.set_visible_devices(physical_devices[0], 'GPU')

    def prepare_data(self):
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            shear_range=0.3,
            zoom_range=0.3,
            horizontal_flip=True,
            fill_mode='nearest')

        validation_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
            self.train_data_dir,
            color_mode='grayscale',
            target_size=(self.IMG_HEIGHT, self.IMG_WIDTH),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True)

        validation_generator = validation_datagen.flow_from_directory(
            self.validation_data_dir,
            color_mode='grayscale',
            target_size=(self.IMG_HEIGHT, self.IMG_WIDTH),
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True)
        
        return train_generator, validation_generator

    def build_model(self):
        model = Sequential()

        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(self.IMG_HEIGHT, self.IMG_WIDTH, 1)))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.1))

        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.1))

        model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.1))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.2))

        model.add(Dense(7, activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print(model.summary())
        
        return model

    def build_model_from_url(self):
            """
            Takkes a TensorFlow Hub URL and creates a Keras Sequential model with it. 


            """

            # Download the pretrained model and save it as keras layer
            feature_extractor_layer = hub.KerasLayer(self.url,
                                                    trainable=False,
                                                    name="feature_extraction_layer",
                                                    input_shape=(self.IMG_HEIGHT, self.IMG_WIDTH, 3))

            # Create our own model
            model = tf.keras.Sequential()
            model.add(Resizing(48, 48))
            model.add(tf.keras.layers.Lambda(lambda x: tf.repeat(x, 3, axis=-1), input_shape=(48, 48, 1)))
            model.add(feature_extractor_layer)
            model.add(Dense(7, activation="softmax", name="output_layer"))
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            return model

    def train(self, epochs=100):
        num_train_imgs = sum([len(files) for r, d, files in os.walk(self.train_data_dir)])
        num_test_imgs = sum([len(files) for r, d, files in os.walk(self.validation_data_dir)])
                            
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, mode='max', verbose=1)

        history = self.model.fit(
            self.train_generator,
            #steps_per_epoch=num_train_imgs // self.batch_size,
            epochs=epochs,
            validation_data=self.validation_generator
            #validation_steps=num_test_imgs // self.batch_size,
            #callbacks=[early_stopping]
        )

        self.model.save('model_test.h5')
        self.plot_model_history(history)
        tf.keras.backend.clear_session()

    @staticmethod
    def plot_model_history(history):
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))

        axs[0].plot(range(1, len(history.history['accuracy']) + 1), history.history['accuracy'])
        axs[0].plot(range(1, len(history.history['val_accuracy']) + 1), history.history['val_accuracy'])
        axs[0].set_title('Model Accuracy')
        axs[0].set_ylabel('Accuracy')
        axs[0].set_xlabel('Epoch')
        axs[0].legend(['train', 'val'], loc='best')

        axs[1].plot(range(1, len(history.history['loss']) + 1), history.history['loss'])
        axs[1].plot(range(1, len(history.history['val_loss']) + 1), history.history['val_loss'])
        axs[1].set_title('Model Loss')
        axs[1].set_ylabel('Loss')
        axs[1].set_xlabel('Epoch')
        axs[1].legend(['train', 'val'], loc='best')

        fig.savefig('train.png')
        plt.show()


train_data_dir = 'data/fer/train/'
validation_data_dir = 'data/fer/test/'

#classifier = EmotionClassifier(train_data_dir, validation_data_dir, "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5")
classifier = EmotionClassifier(train_data_dir, validation_data_dir)
classifier.train(100)