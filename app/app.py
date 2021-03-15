import os
from typing import Optional, Union

import cv2
import numpy as np
from flask import Flask, flash, render_template, request, url_for
from werkzeug.datastructures import FileStorage

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.environ.get('MEDIA_ROOT', '/media/')

ALLOWED_EXTENSIONS = ['.mp4']


def get_file_extension(filename: str):
    name = filename.split('.')
    if len(name) == 1:
        return ''
    return f'.{name[-1]}'


def render_index(state='index'):
    return render_template('index.html', state=state)


def detect_fire(filename: str):
    """
    The algorithm's author's telegram: @TDSbu
    """
    video = cv2.VideoCapture(filename)

    is_detected = False
    while True:
        grabbed, frame = video.read()
        if not grabbed:
            break

        frame = cv2.resize(frame, (960, 540))

        blur = cv2.GaussianBlur(frame, (21, 21), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        lower = [18, 50, 50]
        upper = [35, 255, 255]
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        mask = cv2.inRange(hsv, lower, upper)

        no_red = cv2.countNonZero(mask)

        if int(no_red) > 15000:
            is_detected = True
            break

        # for some reasons, without this check, algorithm does not work
        if cv2.waitKey(1) & 0xFF == ord('q'):  # if 'q' is pressed
            break

    cv2.destroyAllWindows()
    video.release()

    return is_detected


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_index()

    video: Optional[FileStorage] = request.files.get('video')
    if video is None:
        return render_index()

    extension = get_file_extension(video.filename)
    if extension not in ALLOWED_EXTENSIONS:
        return render_index()

    filename = os.path.join(
        app.config['UPLOAD_FOLDER'], f'now_detecting{extension}')
    video.save(filename)

    if detect_fire(filename):
        return render_index('detected')

    return render_index('not_detected')
