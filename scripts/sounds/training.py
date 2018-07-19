import numpy as np
import pandas as pd
import json
import math
import librosa
# from pydub import AudioSegment
from keras.utils import np_utils
from keras.optimizers import Adadelta
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import model_from_json

length = 20


# def preprocess():
#     count = 1
#     for filename in range(1, 16):
#         filename = str(filename)
#         sound = AudioSegment.from_file('./data/raw_sounds/' + filename +
#                                        '.mp3', format='mp3')
#         duration = int(sound.duration_seconds) * 1000
#         print(duration)
#         # sample sound every 5 second
#         if duration <= 5000:
#             sound.export('./data/sounds/' + str(count) + '.mp3', format='mp3')
#             count += 1
#         else:
#             step = 5000
#             for i in range(step, duration, step):
#                 subsound = sound[i - step:i]
#                 subsound.export('./data/sounds/' + str(count) + '.mp3',
#                                 format='mp3')
#                 count += 1


def training_data():
    def wav2mfcc(file_path, max_pad_len=length):
        wave, sr = librosa.load(file_path, mono=True, sr=None)
        wave = wave[::3]
        mfcc = librosa.feature.mfcc(wave, sr=16000)
        print(mfcc.shape)
        if mfcc.shape[1] > max_pad_len:
            mfcc = mfcc[:, :max_pad_len]
        else:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)),
                          mode='constant')
        return mfcc

    jsonfile = {}
    # read labels
    label_file = pd.read_csv('../../data/sounds/training_label.csv',
                             header=None)
    label_file = label_file.values
    for i in range(1, 38):
        file_path = '../../data/sounds/' + str(i) + '.mp3'
        vec = wav2mfcc(file_path).tolist()
        temp = label_file[i]
        assert str(i) == str(temp[0])
        label = temp[4]
        jsonfile[i] = dict(x=vec, y=label)
    with open('../../models/sounds/vec_label.json', 'w') as f:
        json.dump(jsonfile, f)


def get_training_data(dataset):
    # prepare training data
    data_len = len(dataset)
    rate = 0.9
    train_len = math.floor(rate * data_len)
    train_data, train_label = [], []
    test_data, test_label = [], []
    for key, value in dataset.items():
        vec, label = value['x'], value['y']
        i = int(key)
        if i < train_len:
            train_data.append(vec)
            train_label.append(label)
        else:
            test_data.append(vec)
            test_label.append(label)
    train_data, train_label, test_data, test_label = np.array(train_data),\
        np.array(train_label), np.array(test_data), np.array(test_label)
    X_train = train_data.reshape(train_data.shape[0], 20, length, 1)
    X_test = test_data.reshape(test_data.shape[0], 20, length, 1)
    Y_train = np_utils.to_categorical(train_label, 3)
    Y_test = np_utils.to_categorical(test_label, 3)
    print(Y_train)
    return X_train, X_test, Y_train, Y_test


def model_training():
    with open('../../models/sounds/vec_label.json', 'r') as f:
        dataset = json.load(f)
    X_train, X_test, Y_train, Y_test = get_training_data(dataset)
    # init model
    model = Sequential()
    # 1st convolution layer
    model.add(Conv2D(32, kernel_size=(2, 2), strides=(1, 1), activation='relu',
                     input_shape=(20, length, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Dense(3, activation='softmax'))
    # opt = SGD(lr=0.01)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adadelta(), metrics=['accuracy'])
    model.fit(X_train, Y_train, batch_size=30, epochs=10, verbose=1,
              validation_data=(X_test, Y_test))
    # save model
    print('start saving model')
    model_json = model.to_json()
    with open('../../models/sounds/model.json', 'w') as f:
        f.write(model_json)
    model.save_weights('../../models/sounds/model.h5')
    print('finish saving model')


def test():
    with open('../../models/sounds/model.json', 'r') as f:
        model_json = f.read()
    audio_model = model_from_json(model_json)
    audio_model.load_weights('../../models/sounds/model.h5')
    audio_model.compile(loss='categorical_crossentropy',
                        optimizer=Adadelta(), metrics=['accuracy'])
    # process
    max_pad_len = length
    for i in range(1, 30):
        file_path = '../../data/sounds/' + str(i) + '.mp3'
        wave, sr = librosa.load(file_path, mono=True, sr=None)
        wave = wave[::3]
        mfcc = librosa.feature.mfcc(wave, sr=sr)
        if mfcc.shape[1] > max_pad_len:
            mfcc = mfcc[:, :max_pad_len]
        else:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)),
                          mode='constant')
        mfcc = mfcc.reshape(1, 20, max_pad_len, 1)
        res = audio_model.predict(mfcc)[0].tolist()
        print(i, res)


if __name__ == '__main__':
    training_data()
    model_training()
    test()
