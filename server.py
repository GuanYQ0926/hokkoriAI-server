# coding: utf-8
import sys
import io
import os
import werkzeug
import tempfile
import numpy as np
from flask import Flask, request
from flask_restful import Api, Resource, reqparse
import cv2
from keras.models import model_from_json
from keras.optimizers import Adam


# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
app = Flask(__name__)
api = Api(app)


class TextResource(Resource):

    def get(self, parameters):
        print(str(parameters))
        return 'hello'

    def post(self):
        pass


class ImageResource(Resource):

    def get(self, parameters):
        # process image response
        emotions = ['angry', 'disgust', 'fear', 'happy',
                    'sad', 'surprise', 'neutral']
        with open('./models/image/model.json', 'r') as f:
            model_json = f.read()
        model = model_from_json(model_json)
        model.load_weights('./models/image/model.h5')
        print('loaded mode')
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(), metrics=['accuracy'])
        # process image
        image_url = './files/images/temp.png'
        img = cv2.imread(image_url, 0)
        img = cv2.resize(img, (48, 48))
        img = img.reshape((1, 48, 48, 1))
        # predict
        res = model.predict(img)[0]
        emo = np.argmax(res)
        return emotions[emo]

    def post(self):
        parse = reqparse.RequestParser()
        parse.add_argument('images',
                           type=werkzeug.datastructures.FileStorage,
                           location='images')
        data = parse.parse_args()
        if data['images'] == '':
            return {
                'data': '',
                'message': 'No file found',
                'status': 'error'
            }
        images = data['images']
        if images:
            filename = 'temp.png'
            images.save(os.path.join('./files/images', filename))
            return {
                'data': '',
                'message': 'image saved',
                'status': 'success'
            }
        return {
            'data': '',
            'message': 'Error when run',
            'status': 'error'
        }


class AudioResource(Resource):

    def post(self):
        print('in audio processor')
        json_data = request.get_json(force=True)
        text = json_data['text']
        print(text)
        # parse = reqparse.RequestParser()
        # parse.add_argument('audio', type=werkzeug.FileStorage,
        #                    location='files/audio')
        # args = parse.parse_args()
        # stream = args['audio'].stream
        # with tempfile.NamedTemporaryFile(dir='./files/audio/',
        #                                  delete=False) as f:
        #     for chunk in stream.iter_content():
        #         f.write(chunk)
        #     tempfile_path = f.name
        #     os.rename(tempfile_path, './files/audio/temp.wav')
        return 'saved'


class VideoResource(Resource):

    def get(self, parameters):
        pass

    def post(self):
        pass


api.add_resource(TextResource, '/text/<parameters>')
api.add_resource(ImageResource, '/image/<parameters>')
api.add_resource(AudioResource, '/audio')
api.add_resource(VideoResource, '/video/<parameters>')


if __name__ == '__main__':
    port = int(os.getenv("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
