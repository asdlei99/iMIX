# imix demo: demo for imix multimodel framework on vqa and visual dialog task
```
 .----------------.  .----------------.  .----------------.  .----------------. 
| .--------------. || .--------------. || .--------------. || .--------------. |
| |     _____    | || | ____    ____ | || |     _____    | || |  ____  ____  | |
| |    |_   _|   | || ||_   \  /   _|| || |    |_   _|   | || | |_  _||_  _| | |
| |      | |     | || |  |   \/   |  | || |      | |     | || |   \ \  / /   | |
| |      | |     | || |  | |\  /| |  | || |      | |     | || |    > `' <    | |
| |     _| |_    | || | _| |_\/_| |_ | || |     _| |_    | || |  _/ /'`\ \_  | |
| |    |_____|   | || ||_____||_____|| || |    |_____|   | || | |____||____| | |
| |              | || |              | || |              | || |              | |
| '--------------' || '--------------' || '--------------' || '--------------' |
 '----------------'  '----------------'  '----------------'  '----------------' 
```
- IMix is a opensource multimodel framework.
- You can ask imix any **question about your image .**

<br><br>

## Start
```console
python start_demo.py
```
<br><br>

## Model Supports
- [Lxmert]() : [small, medium, large]
- [BlenderBot](https://arxiv.org/abs/2004.13637) : [small, medium, large, xlarge]
- Coming Soon...

<br><br>

## Usage


### 1. Web environment (not terminal)
- You can use image id to manage user-specific history.
- This can be useful when deployed on Facebook messenger or WhatsApp.
- There is a web demo implementation in the `/demo` folder.
- Programming Language:Python
- ML Tools/Libraries: pytorch, Scikit Learn, Numpy
- Web Tools/Libraries: Flask, HTML
<br>
    
#### 1.1. Write your own environment class
- Make your own environment class inherited from `BaseEnv`
- And implement your own `run(model: BaseModel)` method like below.

```python
from typing import Dict
from flask import Flask, render_template
from flask_cors import CORS
from openchat.envs import BaseEnv
from openchat.models import BaseModel


class WebDemoEnv(BaseEnv):

    def __init__(self):
        super().__init__()
        self.app = Flask(__name__)
        CORS(self.app)

    def run(self, model: BaseModel):

        @self.app.route("/")
        def index():
            return render_template("index.html", title=model.name)

        @self.app.route('/send/<image_id>/<text>', methods=['GET'])
        def send(image_id, text: str) -> Dict[str, str]:

            if text in self.keywords:
                # Format of self.keywords dictionary
                # self.keywords['/exit'] = (exit_function, 'good bye.')

                _out = self.keywords[text][1]
                # text to print when keyword triggered

                self.keywords[text][0](image_id, text)
                # function to operate when keyword triggered

            else:
                _out = model.predict(image_id, text)

            return {"output": _out}

        self.app.run(host="0.0.0.0", port=5050)
```
<br>

### 2. Start to run application.
```python
from openchat import OpenChat
from demo.web_demo_env import WebDemoEnv

OpenChat(model="vqa_model", env=WebDemoEnv())
```
<br><br>

### 3. Additional Options
#### 3.1. Add custom Keywords

- You can add new manual keyword such as `.exit`, `.clear`, 
- call the `self.add_keyword('.new_keyword', 'message to print', triggered_function)'` method.
- `triggered_function` should be form of `function(user_id:str, text:str)`

```python
from openchat.envs import BaseEnv


class YourOwnEnv(BaseEnv):
    
    def __init__(self):
        super().__init__()
        self.add_keyword(".new_keyword", "message to print", self.function)

    def function(self, user_id: str, text: str):
        """do something !"""
        
```
<br><br>

#### 3.3. Check histories
- You can check all dialogue history using `self.histories` for **visual dialog**
```python
from openchat.envs import BaseEnv


class YourOwnEnv(BaseEnv):
    
    def __init__(self):
        super().__init__()
        print(self.histories)
```
```
{
    image_id1 : {'user': [] , 'bot': []},
    image_id2 : {'user': [] , 'bot': []},
    ...more...
    image_idn : {'user': [] , 'bot': []},
}
```
<br>

#### 3.4. Clear histories
- You can clear all dialogue histories
```python
from flask import Flask
from openchat.envs import BaseEnv
from openchat.models import BaseModel

class YourOwnEnv(BaseEnv):
    
    def __init__(self):
        super().__init__()
        self.app = Flask(__name__)

    def run(self, model: BaseModel):
        
        @self.app.route('/send/<image_id>/<text>', methods=['GET'])
        def send(image_id, text: str) -> Dict[str, str]:
            
            self.clear(image_id, text)
            # clear all histories ! 

```

<br><br>
