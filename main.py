import os
import sys
import io
import json
import MeCab
import tempfile
import librosa
import cv2
import numpy as np
from keras.models import model_from_json
from keras.optimizers import Adam


from flask import Flask, request, abort
from linebot import (
    LineBotApi, WebhookHandler
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
    ImageMessage, VideoMessage, AudioMessage,
    LocationMessage, StickerMessage, FileMessage,
    ButtonsTemplate, URITemplateAction, TemplateSendMessage
)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
app = Flask(__name__)

with open('./data/environment.json', 'r') as f:
    environ = json.load(f)
channel_secret = environ['channel_secret']
channel_access_token = environ['channel_access_token']
if channel_secret is None:
    print('Specify LINE_CHANNEL_SECRET as environment variable.')
    sys.exit(1)
if channel_access_token is None:
    print('Specify LINE_CHANNEL_ACCESS_TOKEN as environment variable.')
    sys.exit(1)

line_bot_api = LineBotApi(channel_access_token)
handler = WebhookHandler(channel_secret)


@app.route("/")
def hello_world():
    return '<a href="http://www.viz.media.kyoto-u.ac.jp/">Koyamada Lab</a>'


@app.route("/test")
def test():
    return '<p>this is test html</p>'


@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)
    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return 'OK'


# process text message
@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    send_message = get_text_response(event)
    line_bot_api.reply_message(
        event.reply_token,
        send_message
    )


# process image message
@handler.add(MessageEvent, message=ImageMessage)
def handle_image_message(event):
    message_content = line_bot_api.get_message_content(event.message.id)
    with tempfile.NamedTemporaryFile(dir='./tempimage/', delete=False) as f:
        for chunk in message_content.iter_content():
            f.write(chunk)
        tempfile_path = f.name
        os.rename(tempfile_path, './tempimage/temp.jpg')
    send_message = get_image_response('./tempimage/temp.jpg')
    line_bot_api.reply_message(
        event.reply_token,
        send_message
    )


# process video message
@handler.add(MessageEvent, message=VideoMessage)
def handle_video_message(event):
    message_content = line_bot_api.get_message_content(event.message.id)
    with tempfile.NamedTemporaryFile(dir='./tempvideo/', delete=False) as f:
        for chunk in message_content.iter_content():
            f.write(chunk)
        tempfile_path = f.name
        os.rename(tempfile_path, './tempvideo/temp.mp4')
    send_message = get_video_response('./tempvideo/temp.mp4')
    line_bot_api.reply_message(
        event.reply_token,
        send_message
    )


# process audio message
@handler.add(MessageEvent, message=AudioMessage)
def handle_audio_message(event):
    message_content = line_bot_api.get_message_content(event.message.id)
    with tempfile.NamedTemporaryFile(dir='./tempaudio/', delete=False) as f:
        for chunk in message_content.iter_content():
            f.write(chunk)
        tempfile_path = f.name
        os.rename(tempfile_path, './tempaudio/temp.wav')
    send_message = get_audio_response('./tempaudio/temp.wav')
    line_bot_api.reply_message(
        event.reply_token,
        send_message
    )


# process other message
@handler.add(MessageEvent, message=(LocationMessage, StickerMessage,
                                    FileMessage))
def handle_other_message(event):
    text = 'unknown message type'
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=text)
    )


def get_text_response(event):
    text = event.message.text
    if text == 'egrid':
        buttons_template = \
            ButtonsTemplate(title='Interview', text='E-Grid Visualization',
                            actions=[URITemplateAction(
                                label='Go to E-Grid',
                                uri='https://egrid.jp/projects')])
        response_message = TemplateSendMessage(alt_text='E-grid link',
                                               template=buttons_template)
        return response_message
    elif text == 'upload':
        buttons_template = \
            ButtonsTemplate(title='Upload', text='Upload data to database',
                            actions=[URITemplateAction(
                                label='Go to data form',
                                uri='https://yuqingguan.top/upload')])
        response_message = TemplateSendMessage(alt_text='upload link',
                                               template=buttons_template)
        return response_message
    else:
        profile = line_bot_api.get_profile(event.source.user_id)
        reply_text = profile.display_name + ' : '
        tagger = MeCab.Tagger('mecabrc')
        tagger.parse('')
        node = tagger.parseToNode(text)
        while node:
            word = node.surface
            reply_text += (word + ' ')
            node = node.next
        return TextSendMessage(reply_text)


def get_image_response(image_url):
    emotions = ['angry', 'disgust', 'fear', 'happy',
                'sad', 'surprise', 'neutral']
    # load model
    with open('./app/ml/images/model.json', 'r') as f:
        model_json = f.read()
    model = model_from_json(model_json)
    model.load_weights('./app/ml/images/model.h5')
    print('loaded mode from dist')
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(), metrics=['accuracy'])
    # process image
    img = cv2.imread(image_url, 0)
    img = cv2.resize(img, (48, 48))
    img = img.reshape((1, 48, 48, 1))
    # predict
    res = model.predict(img)[0]
    emo = np.argmax(res)
    return TextSendMessage(emotions[emo])


def get_video_response(video_url):
    emotions = ['angry', 'disgust', 'fear', 'happy',
                'sad', 'surprise', 'neutral']
    # load model
    with open('./app/ml/images/model.json', 'r') as f:
        model_json = f.read()
    model = model_from_json(model_json)
    model.load_weights('./app/ml/images/model.h5')
    print('loaded mode from dist')
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(), metrics=['accuracy'])
    cap = cv2.VideoCapture(video_url)
    if not cap.isOpened():
        return TextSendMessage(text='Error')
    count = 0
    emolist = [0 for _ in range(7)]
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if count % 3 == 0:
                img = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                img = cv2.resize(img, (48, 48))
                img = img.reshape((1, 48, 48, 1))
                res = model.predict(img)[0]
                emolist[np.argmax(res)] += 1
            count += 1
        else:
            break
    cap.release()
    res = ''
    for i, v in enumerate(emolist):
        if v != 0:
            res += emotions[i] + ':' + str(v) + ' '
    return TextSendMessage(res)


def get_audio_response(audio_url):
    # max_pad_len = 11
    # wave, sr = librosa.load(audio_url, mono=True, sr=None)
    # wave = wave[::3]
    # mfcc = librosa.feature.mfcc(wave, sr=16000)
    # pad_width = max_pad_len - mfcc.shape[1]
    # mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    # audio features
    return TextSendMessage('saved')


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
