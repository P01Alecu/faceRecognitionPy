import cv2 
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('model_file.h5')

#test the accuracy with the test set
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import os
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
#config = ConfigProto()
#config.gpu_options.allow_growth = True
#session = InteractiveSession(config=config)

#physical_devices = tf.config.list_physical_devices('GPU')
#print(physical_devices)
#tf.config.set_visible_devices(physical_devices[0], 'GPU')

validation_data_dir = 'data/test/'
validation_datagen = ImageDataGenerator(rescale = 1./255)
validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    color_mode='grayscale',
    target_size=(48,48),
    batch_size=32,
    class_mode='categorical',
    shuffle=True)
model.evaluate(validation_generator)

#test with 1 photo
"""
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

label_dict = {0: 'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Neutral', 5:'Sad', 6:'Surprise'}

#frame = cv2.imread("faces-small.jpg")
#frame = cv2.imread("sad-face.jpeg")
#frame = cv2.imread("happyFamily.jpg")
#frame = cv2.imread("disgusted.jpg")
frame = cv2.imread("angry.jpeg")

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = faceDetect.detectMultiScale(gray, 1.3, 3)
for x, y, w, h in faces:
    sub_face_img = gray[y : y + h, x : x + w]
    resized = cv2.resize(sub_face_img, (48, 48))
    normalize = resized / 255.0
    reshaped = np.reshape(normalize, (1, 48, 48, 1))
    result = model.predict(reshaped)
    label = np.argmax(result, axis = 1)[0]
    
    print(label)

    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
    cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
    cv2.putText(frame, label_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255))

cv2.imshow("Frame", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""