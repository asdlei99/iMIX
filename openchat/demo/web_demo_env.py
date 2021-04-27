from typing import Dict
from flask import Flask, render_template, request
from flask_cors import CORS

from openchat.envs import BaseEnv
from openchat.models import BaseModel
import glob
import os
from werkzeug.utils import secure_filename


IMAGE_PATH = '/home/zyj/openchat_v2/openchat/demo/static/image'
questions = []

class WebDemoEnv(BaseEnv):

    def __init__(self):
        super().__init__()
        self.app = Flask(__name__)
        CORS(self.app) #解决跨域问题

    def run(self, model: BaseModel):

        @self.app.route("/", methods=["GET","POST"])
        def index():

            return render_template("index.html", title=model.name)

        @self.app.route('/image', methods=['POST'])
        def save_image():
            f = request.files['file']
            print(f)
            fname = secure_filename(f.filename)
            f.save(IMAGE_PATH + '/' + fname)
            return {'output': 'ok! let\'s start'}

        @self.app.route('/send/<imageName>/<text>', methods=['GET'])
        def send(imageName, text: str) -> Dict[str, str]:

            if text in self.keywords:
                # Format of self.keywords dictionary
                # self.keywords['/exit'] = (exit_function, 'good bye.')

                _out = self.keywords[text][1]
                # text to print when keyword triggered

                self.keywords[text][0](imageName, text)
                # function to operate when keyword triggered

            else:
                outputs = model.predict(imageName, text)


            return {"output": outputs}

        self.app.run(host="0.0.0.0", port=5050)
