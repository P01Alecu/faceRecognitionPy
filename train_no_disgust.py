import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
import os



from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[0], 'GPU')

train_data_dir = 'dataNDisgust/train/'
validation_data_dir = 'dataNDisgust/test/'

IMG_HEIGHT = 48
IMG_WIDTH = 48
batch_size = 32

# data augmentation
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range=30,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale = 1./255)

# generator
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    color_mode='grayscale',
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical',
    shuffle=True)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    color_mode='grayscale',
    target_size=(48,48),
    batch_size=32,
    class_mode='categorical',
    shuffle=True)

class_labels=['Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

img, label = train_generator.__next__()


model = Sequential()


# adauga layerele 
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))

model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(6, activation='softmax'))   # 7 pentru ca avem 7 categorii

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())



#### aici trainuim modelul (numaram inputuri)

train_path = "dataNDisgust/train/"
test_path = "dataNDisgust/test/"

num_train_imgs = 0 
for root, dirs, files in os.walk(train_path):
    num_train_imgs += len (files)

num_test_imgs = 0
for root, dirs, files in os.walk(test_path):
    num_test_imgs += len(files)    

# aici se face trainul efectiv
epochs = 100

# better use earlyStop > checkpoint
#checkpoint = ModelCheckpoint("best_model.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_accuracy', patience = 15, mode = 'max', verbose = 1)

history = model.fit(train_generator,
                    steps_per_epoch = num_train_imgs//32,
                    epochs = epochs,
                    validation_data = validation_generator,
                    validation_steps = num_test_imgs//32,
                    callbacks = [early_stopping])

model.save('model_early.h5')


# free the memory used by the model
tf.keras.backend.clear_session()