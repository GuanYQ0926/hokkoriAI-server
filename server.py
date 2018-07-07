# coding: utf-8
import sys
import io
import os
import librosa
import numpy as np
from flask import Flask, request
from flask_restful import Api, Resource
from keras.models import model_from_json
from keras.optimizers import Adadelta


os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
app = Flask(__name__)
api = Api(app)


# load model in advance
with open('./models/sounds/model.json', 'r') as f:
    model_json = f.read()
audio_model = model_from_json(model_json)
audio_model.load_weights('./models/sounds/model.h5')
print('model loaded')
audio_model.compile(loss='categorical_crossentropy', optimizer=Adadelta(),
                    metrics=['accuracy'])


class TextResource(Resource):

    def get(self, parameters):
        print(str(parameters))
        return 'hello'


class AudioResource(Resource):

    def post(self):
        # save audio file
        file_path = './tempfiles/audio/temp.m4a'
        file = request.files['file']
        file.save(file_path)
        # process
        max_pad_len = 500
        wave, sr = librosa.load(file_path, mono=True, sr=None)
        mfcc = librosa.feature.mfcc(wave, sr=sr)
        if mfcc.shape[1] > max_pad_len:
            mfcc = mfcc[:, :max_pad_len]
        else:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)),
                          mode='constant')
        mfcc = mfcc.reshape(1, 20, max_pad_len, 1)
        result = audio_model.predict(mfcc)[0]
        print('======', result)
        return result


api.add_resource(TextResource, '/text/<parameters>')
api.add_resource(AudioResource, '/audio')


if __name__ == '__main__':
    port = int(os.getenv("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
