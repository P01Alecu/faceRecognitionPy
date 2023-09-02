import cv2 
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # hide tf warnings
import tensorflow as tf
from tensorflow.keras.models import load_model
import getFilesForTest

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

class ModelTester:
    def __init__(self, model_path, target_size):
        self.model = load_model(model_path)
        self.target_size = target_size
        
        self.face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.label_dict = {0: 'Angry', 1:'Disgust', 2:'Fear', 3:'Happy', 4:'Neutral', 5:'Sad', 6:'Surprise'}

    def evaluate(self, validation_data_dir):
        validation_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
            validation_data_dir,
            color_mode='rgb',
            target_size=self.target_size,
            batch_size=32,
            class_mode='categorical',
            shuffle=True)
        loss, acc = self.model.evaluate(validation_generator)
        print(f'Loss: {loss}, Accuracy: {acc}')
        tf.keras.backend.clear_session()

    def evaluate_personal_data(self, data_dir):
        list_imagesWlabel = getFilesForTest.list_files_in_folder(data_dir)
        labels = []
        images = []
        predictions = []
        for image_wLabel in list_imagesWlabel:
            images.append(image_wLabel[0])
            tempPred = self.predict_image(image_wLabel[1] + '/' + image_wLabel[0], 1)
            if len(tempPred) == 0:
                del images[-1]
                continue
            predictions.append(tempPred[0])
            labels.append(getFilesForTest.get_substring_before_last_slash(image_wLabel[1]))
            if labels[-1] in self.label_dict.values():
                for cheie, valoare in self.label_dict.items():
                    if valoare == labels[-1]:
                        labels[-1] = cheie
                        break

        # Calculați matricea de confuzie
        confusion_mat = confusion_matrix(labels, predictions)

        # Normalizați matricea de confuzie
        confusion_matrix_normalized = confusion_mat.astype('float') / confusion_mat.sum(axis=1, keepdims=True)

        # Creați o vizualizare grafică a matricei de confuzie normalizate
        plt.imshow(confusion_matrix_normalized, cmap='Blues')
        plt.title('Matrice de confuzie normalizată')
        plt.xlabel('Predictii')
        plt.ylabel('Etichetă reală')
        plt.xticks(np.arange(len(self.label_dict)), self.label_dict)
        plt.yticks(np.arange(len(self.label_dict)), self.label_dict)

        # Adăugați proporțiile normalizate la matrice
        for i in range(confusion_matrix_normalized.shape[0]):
            for j in range(confusion_matrix_normalized.shape[1]):
                plt.text(j, i, "{:.2f}".format(confusion_matrix_normalized[i, j]), color='white', ha='center', va='center')

        plt.colorbar()  # adăugați o bară de culoare pentru referință
        plt.show()
        
    def predict_image(self, image_path, mode = 0):
        if not self.model:
            raise Exception("Modelul nu a fost incarcat.")
        labels = []
        frame = cv2.imread(image_path)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.face_detect.detectMultiScale(gray, 1.3, 3)
        for x, y, w, h in faces:
            sub_face_img = gray[y : y + h, x : x + w]
            resized = cv2.resize(sub_face_img, self.target_size)
            normalize = resized / 255.0
            reshaped = np.expand_dims(normalize, axis=0)
            result = self.model.predict(reshaped)
            print(result)
            label = np.argmax(result, axis=1)[0] # can be deleted and replaced with the one below
            labels.append(np.argmax(result, axis=1)[0])
            if mode == 0:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
                cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
                cv2.putText(frame, self.label_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255))
        if mode == 0:
            cv2.imshow("Frame", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            tf.keras.backend.clear_session()
        return labels

    def predict_web(self):
        video = cv2.VideoCapture(0)
        while True:
            ret, frame = video.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # faces o sa contina coordonatele fiecarei fete din imaginea incarcata
            faces = self.face_detect.detectMultiScale(gray, 1.3, 3)
            # se ia fiecare fata in parte si se face predictia
            for x, y, w, h in faces:
                sub_face_img = gray[y : y + h, x : x + w]
                resized = cv2.resize(sub_face_img, self.target_size)
                normalize = resized / 255.0
                reshaped = np.expand_dims(normalize, axis=0)
                #reshaped = np.reshape(normalize, (1, *self.target_size, 1))
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

    

tester = ModelTester('complete.h5', (197,197))
#labels = tester.predict_image('data/happyFamily.jpg')
tester.evaluate_personal_data('data/app/')


#tester.predict_web()