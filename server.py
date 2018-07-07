# coding: utf-8
import sys
import io
import os
from flask import Flask, request
from flask_restful import Api, Resource


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
app = Flask(__name__)
api = Api(app)


class TextResource(Resource):

    def get(self, parameters):
        print(str(parameters))
        return 'hello'


class AudioResource(Resource):

    def post(self):
        print('in audio processor')
        file = request.files['file']
        file.save('./tempfiles/audio/temp.m4a')
        return 'saved'


api.add_resource(TextResource, '/text/<parameters>')
api.add_resource(AudioResource, '/audio')


if __name__ == '__main__':
    port = int(os.getenv("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
