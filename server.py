# coding: utf-8
import sys
import io
import os
import librosa
import numpy as np
from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from keras.models import model_from_json
from keras.optimizers import Adadelta


os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
app = Flask(__name__)
api = Api(app)


class TextResource(Resource):

    def get(self, parameters):
        return str(parameters)


class AudioResource(Resource):

    def post(self):
        # load model in advance
        with open('./models/sounds/model.json', 'r') as f:
            model_json = f.read()
        audio_model = model_from_json(model_json)
        audio_model.load_weights('./models/sounds/model.h5')
        print('model loaded')
        audio_model.compile(loss='categorical_crossentropy',
                            optimizer=Adadelta(), metrics=['accuracy'])
        # save audio file
        print('audio request in coming')
        file_path = './tempfiles/audio/temp.mp3'
        file = request.files['file']
        file.save(file_path)
        print('audio file is temporally saved')
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
        res = audio_model.predict(mfcc)[0].tolist()
        print('prediction end')
        return jsonify(fussy=res[0], hungry=res[1], pain=res[2])


api.add_resource(TextResource, '/text/<parameters>')
api.add_resource(AudioResource, '/audio')


if __name__ == '__main__':
    port = int(os.getenv("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
