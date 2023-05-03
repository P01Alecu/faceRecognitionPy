import cv2 
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # hide tf warnings
from tensorflow.keras.models import load_model

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

class ModelTester:
    def __init__(self, model_path, validation_data_dir, target_size, batch_size, modelUsed = 0):
        self.modelUsed = modelUsed # 0 - FER     1 - CK+     2 - affectNET

        self.model = load_model(model_path)
        self.validation_data_dir = validation_data_dir
        self.target_size = target_size
        self.batch_size = batch_size
        self.validation_datagen = ImageDataGenerator(rescale=1./255)

        self.face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        if(modelUsed == 1):
            # for CK+ dataset
            self.label_dict = {0: 'Angry', 1:'contempt', 2:'disgust', 3:'fear', 4:'happy', 5:'Sad', 6:'Surprise'}
        elif(modelUsed == 2):
            # for AffectNet dataset
            self.label_dict = {0: 'surprise', 1:'fear', 2:'neutral', 3:'sad', 4:'disgust', 5:'contempt', 6:'happy', 7:'anger'}
        else:
            #FER dataset
            self.label_dict = {0: 'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Neutral', 5:'Sad', 6:'Surprise'}
        print(self.label_dict)
        self.fix_gpu()

    def fix_gpu(self):
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.set_visible_devices(physical_devices[0], 'GPU')

    def evaluate(self):
        validation_generator = self.validation_datagen.flow_from_directory(
            self.validation_data_dir,
            color_mode='grayscale',
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True)
        loss, acc = self.model.evaluate(validation_generator)
        print(f'Loss: {loss}, Accuracy: {acc}')
        tf.keras.backend.clear_session()

        
    def predict_image(self, image_path):
        if not self.model:
            raise Exception("Modelul nu a fost incarcat.")
        frame = cv2.imread(image_path)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detect.detectMultiScale(gray, 1.3, 3)
        for x, y, w, h in faces:
            sub_face_img = gray[y : y + h, x : x + w]
            resized = cv2.resize(sub_face_img, self.target_size)
            normalize = resized / 255.0
            reshaped = np.reshape(normalize, (1, *self.target_size, 1))
            result = self.model.predict(reshaped)
            print(result)
            label = np.argmax(result, axis=1)[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
            cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
            cv2.putText(frame, self.label_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255))
        cv2.imshow("Frame", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        tf.keras.backend.clear_session()
    
    def predict_image_with_output(self, input_image):
        frame = input_image.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detect.detectMultiScale(gray, 1.3, 3)
        for x, y, w, h in faces:
            sub_face_img = gray[y : y + h, x : x + w]
            resized = cv2.resize(sub_face_img, self.target_size)
            normalize = resized / 255.0
            reshaped = np.reshape(normalize, (1, *self.target_size, 1))
            result = self.model.predict(reshaped)
            label = np.argmax(result, axis=1)[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
            cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
            cv2.putText(frame, self.label_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255))
        return frame

    def predict_web(self):
        video = cv2.VideoCapture(0)
        while True:
            ret, frame = video.read()
            # se aplica filtru gray
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # faces o sa contina coordonatele fiecarei fete din imaginea incarcata
            faces = self.face_detect.detectMultiScale(gray, 1.3, 3)
            # se ia fiecare fata in parte si se face predictia
            for x, y, w, h in faces:
                sub_face_img = gray[y : y + h, x : x + w]
                resized = cv2.resize(sub_face_img, self.target_size)
                normalize = resized / 255.0
                reshaped = np.reshape(normalize, (1, *self.target_size, 1))
                result = self.model.predict(reshaped)
                label = np.argmax(result, axis = 1)[0]

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
                cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
                cv2.putText(frame, self.label_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255))

            cv2.imshow("Frame", frame)
            k = cv2.waitKey(1)
            if k == ord('q'):
                break
        video.release()
        cv2.destroyAllWindows()
        tf.keras.backend.clear_session()

#tester = ModelTester('model_file_100epochs.h5', 'data/test/', (48,48), 32, 0)
tester = ModelTester('affectNet_v1.h5', 'data/fer/test/', (96,96), 32, 2)
tester.predict_image('data/happyFamily.jpg')
#tester.evaluate()
#tester.predict_image('data/sad-face.jpeg')
#tester.predict_web()