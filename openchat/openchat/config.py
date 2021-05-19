import os
import json
root_path = '/home/zyj/openchat_v2/openchat'
image_path = '/home/zyj/openchat_v2/openchat/demo/static/image/'
lxmert_weight_path = os.path.join(root_path, 'openchat/model_pth/lxmrt.pth')
detect_weight_path = os.path.join(root_path, 'openchat/model_pth/detect.pth')

answer_table = json.load(open('/home/datasets/mix_data/lxmert/vqa/trainval_label2ans.json'))
