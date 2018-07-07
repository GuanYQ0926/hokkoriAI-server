import os.path
import numpy as np
import pandas as pd
# import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.models import model_from_json

# tf.python.control_flow_ops = tf


def data_processing(images_file_name='../../data/fer2013/facial_data_images',
                    labels_file_name='../../data/fer2013/facial_data_labels'):
    csv_file = pd.read_csv('../../data/fer2013/fer2013.csv')
    data = csv_file.values  # emotion | pixel | usage
    pixels = data[:, 1]
    labels = data[:, 0]
    imgs = np.zeros((pixels.shape[0], 48 * 48))
    for ix in range(imgs.shape[0]):
        p = pixels[ix].split(' ')
        for iy in range(imgs.shape[1]):
            imgs[ix, iy] = int(p[iy])
    np.save(images_file_name, imgs)
    np.save(labels_file_name, labels)
    return imgs, labels


def model_training(
        images_file_name='../../data/fer2013/facial_data_images.npy',
        labels_file_name='../../data/fer2013/facial_data_labels.npy'):
    if os.path.isfile(images_file_name) and os.path.isfile(labels_file_name):
        print('file existed')
        images = np.load(images_file_name)
        labels = np.load(labels_file_name)
    else:
        print('file not existed')
        images, labels = data_processing()
    images_train = images[:28710, :]
    images_test = images[28710: 32300, :]
    images_train = images_train.reshape((images_train.shape[0], 48, 48, 1))
    images_test = images_test.reshape(
        (images_test.shape[0], 48, 48, 1))
    labels = np_utils.to_categorical(labels, 7)
    labels_train = labels[:28710]
    labels_test = labels[28710:32300]
    print('training & test images: ', images_train.shape, images_test.shape)
    print('training & test labels: ', labels_train.shape, labels_test.shape)
    # init model
    model = Sequential()
    # 1st convolution layer
    model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48, 48, 1)))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))

    # 2nd convolution layer
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

    # 3rd convolution layer
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Flatten())

    # fully connected neural networks
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(7, activation='softmax'))
    print(model.input_shape)

    # training
    gen = ImageDataGenerator()
    train_generator = gen.flow(images_train, labels_train, batch_size=128)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(), metrics=['accuracy'])
    model.fit_generator(train_generator, steps_per_epoch=128, epochs=30)

    # evaluation
    train_score = model.evaluate(images_train, labels_train, verbose=0)
    print('Train loss: ', train_score[0])
    print('Train accuracy: ', 100 * train_score[1])
    test_score = model.evaluate(images_test, labels_test, verbose=0)
    print('Test loss: ', test_score[0])
    print('Test accuracy: ', 100 * test_score[1])

    # save model
    print('start saving model')
    model_json = model.to_json()
    with open('model.json', 'w') as f:
        f.write(model_json)
    model.save_weights('model.h5')
    print('finish saving model')


def test():
    with open('model.json', 'r') as f:
        model_json = f.read()
    loaded_model = model_from_json(model_json)
    loaded_model.load_weights('model.h5')
    print('loaded mode from dist')
    loaded_model.compile(loss='categorical_crossentropy',
                         optimizer=Adam(), metrics=['accuracy'])
    images_file_name = '../../data/fer2013/facial_data_images.npy'
    labels_file_name = '../../data/fer2013/facial_data_labels.npy'
    images = np.load(images_file_name)
    labels = np.load(labels_file_name)
    images_test = images[28710: 32300, :]
    images_test = images_test.reshape(
        (images_test.shape[0], 48, 48, 1))
    labels = np_utils.to_categorical(labels, 7)
    labels_test = labels[28710:32300]
    score = loaded_model.evaluate(images_test, labels_test, verbose=0)
    print('Test loss: ', score[0])
    print('Test accuracy: ', 100 * score[1])


if __name__ == '__main__':
    # model_training()
    test()
