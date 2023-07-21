# 필요한 라이브러리 import
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import tensorflow_addons as tfa
import matplotlib.pyplot as plt

import os
import cv2
import glob

# 학습 데이터 로드 및 전처리
root_folder = 'mvtec_anomaly_detection/'
folders = os.listdir(root_folder)

images = []
labels = []

for folder in folders:
    folder_path = os.path.join(root_folder, folder)
    if os.path.isdir(folder_path):
        subset_path = os.path.join(folder_path, 'test')
        subfolders = os.listdir(subset_path)
        for subfolder in subfolders:
            subfolder_path = os.path.join(subset_path, subfolder)
            files = glob.glob(os.path.join(subfolder_path, '*'))
            for file_path in files:
                img = Image.open(file_path)
                img = img.resize((128, 128))
                img = np.array(img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img / 255.0
                images.append(img)
                if subfolder == 'good':
                    labels.append(folder + '_good')
                else:
                    labels.append(folder + '_bad')

images = np.array(images)
labels = np.array(labels)

# 데이터 분할
unique_labels = np.unique(labels)
label_mapping = {label: i for i, label in enumerate(unique_labels)}
mapped_labels = np.array([label_mapping[l] for l in labels])
label_len = len(unique_labels)

category_label = tf.keras.utils.to_categorical(mapped_labels, label_len)
x_train, x_test, y_train, y_test = train_test_split(images, category_label, test_size=0.2, random_state=1)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=1)

# 모델 생성 및 컴파일
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(128, 128, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(label_len, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy', tfa.metrics.F1Score(num_classes=label_len, average='macro')])

# 모델 학습
checkpointer = ModelCheckpoint(filepath='./model/ad.h5', monitor='val_loss', verbose=1, save_best_only=True)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=20)
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=50, callbacks=[early_stopping_callback, checkpointer])

### Loss(Train, Test) Graph
y_vloss = history.history['val_loss']
y_loss = history.history['loss']

x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.', c="red", label='Test_loss')
plt.plot(x_len, y_loss, marker='.', c="blue", label='Train_loss')

plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()


### Accuracy(Train, Test) Graph
y_vacc = history.history['val_accuracy']
y_acc = history.history['accuracy']

plt.plot(x_len, y_vacc, marker='.', c="red", label='Test_acc')
plt.plot(x_len, y_acc, marker='.', c="blue", label='Train_acc')

plt.grid()
plt.xlabel('epoch')
plt.ylabel('acc')
plt.show()


### F1-Score(Train, Test) Graph
y_vaf = history.history['val_f1_score']
y_f = history.history['f1_score']

plt.plot(x_len, y_vaf, marker='.', c="red", label='Testset_acc')
plt.plot(x_len, y_f, marker='.', c="blue", label='Trainset_acc')

plt.grid()
plt.xlabel('epoch')
plt.ylabel('F1-Score')
plt.show()

def get_label_from_mapping(mapping, target_value):
    for key, value in mapping.items():
        if value == target_value:
            return key
    return None

# 이미지 예측
def predict(img_path):
    model = load_model('./model/ad.h5')
    img = Image.open(img_path)
    img = img.resize((128, 128))
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    predicted_label = np.argmax(prediction)
    label = get_label_from_mapping(label_mapping, predicted_label)
    return label