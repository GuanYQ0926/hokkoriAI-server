# Create your first MLP in Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy as np
import librosa


def training():
    # fix random seed for reproducibility
    np.random.seed(7)

    cry_data_list, song_data_list, cob_data_list = [], [], []
    cry_file_list = ['../../data/sounds/raw_sounds/crying_sounds/'+str(i)+'.m4a'
                     for i in range(1, 16)]
    song_file_list = ['../../data/sounds/line_data/BASE.m4a']
    cob_file_list = ['../../data/sounds/line_data/LINE1.m4a',
                     '../../data/sounds/line_data/LINE3.m4a']
    for fl in cry_file_list:
        data, _ = librosa.load(fl, sr=16000)
        cry_data_list.append(data)
    for fl in song_file_list:
        data, sr = librosa.load(fl, sr=16000)
        data = data[:80*sr]
        song_data_list.append(data)
    for fl in cob_file_list:
        data, sr = librosa.load(fl, sr=16000)
        data = data[:80*sr]
        cob_data_list.append(data)

    # 0.002 * 16000 = 32
    X, Y = [], []
    for cry_data in cry_data_list:
        for i in range(32, len(cry_data), 32):
            X.append(cry_data[i-32: i])
            Y.append(0)
    for cob_data in cob_data_list:
        for i in range(32, len(cob_data), 32):
            X.append(cob_data[i-32: i])
            Y.append(1)
    for song_data in song_file_list:
        for i in range(32, len(song_data), 32):
            X.append(cry_data[i-32: i])
            Y.append(2)
    assert len(X) == len(Y)
    X = np.array(X)
    Y = np.array(Y)
    print(X.shape, Y.shape)
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=32, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    # Fit the model
    model.fit(X, Y, epochs=100, batch_size=10)
    # evaluate the model
    scores = model.evaluate(X, Y)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    # save model
    print('start saving model')
    model_json = model.to_json()
    with open('../../models/sounds/model_mix.json', 'w') as f:
        f.write(model_json)
    model.save_weights('../../models/sounds/model_mix.h5')
    print('finish saving model')


def test():
    with open('../../models/sounds/model_mix.json', 'r') as f:
        model_json = f.read()
    audio_model = model_from_json(model_json)
    audio_model.load_weights('../../models/sounds/model_mix.h5')
    audio_model.compile(loss='binary_crossentropy', optimizer='adam',
                        metrics=['accuracy'])
    fl = '../../data/sounds/line_data/LINE2.m4a'
    fl = '../../data/sounds/raw_sounds/crying_sounds/3.m4a'
    fl = '../../data/sounds/line_data/test.m4a'
    test_data, _ = librosa.load(fl, sr=16000)
    X, Y = [], []
    for i in range(32, len(test_data), 32):
        X.append(test_data[i-32: i])
        Y.append(1)
    X = np.array(X)
    Y = np.array(Y)
    # for tx in X:
    #     print(tx)
    #     res = audio_model.predict(tx)
    #     print(res)
    score = audio_model.evaluate(X, Y, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


if __name__ == '__main__':
    test()
