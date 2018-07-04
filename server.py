# coding: utf-8
import sys
import io
import os
import werkzeug
import tempfile
from flask import Flask, request
from flask_restful import Api, Resource, reqparse


# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
app = Flask(__name__)
api = Api(app)


class TextResource(Resource):

    def get(self, parameters):
        print(str(parameters))
        return 'hello'


class AudioResource(Resource):

    def post(self):
        print('in audio processor')
        # parse = reqparse.RequestParser()
        # parse.add_argument('audio', type=werkzeug.FileStorage,
        #                    location='files/audio')
        # data = parse.parse_args()
        # stream = data['audio']
        json_data = request.get_json(force=True)
        stream = json_data['audio']
        with tempfile.NamedTemporaryFile(dir='./files/audio/',
                                         delete=False) as f:
            for chunk in stream.iter_content():
                f.write(chunk)
            tempfile_path = f.name
            os.rename(tempfile_path, './files/audio/temp.wav')
        return 'saved'


api.add_resource(TextResource, '/text/<parameters>')
api.add_resource(AudioResource, '/audio')


if __name__ == '__main__':
    port = int(os.getenv("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
