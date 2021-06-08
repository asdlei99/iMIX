import os
import json

root_path = '/home/datasets/mix_data/openchat'

image_path = os.path.join(root_path, 'static/image/')
detect_weight_path = os.path.join(root_path, 'model_pth/detect.pth')

lxmert_weight_path = os.path.join(root_path, 'model_pth/lxmert_vqa.pth')
# vilbert_weight_path = os.path.join(root_path, 'model_pth/vilbert_vqa.pth')
# oscar_weight_path = os.path.join(root_path, 'model_pth/oscar_vqa.pth')
# vinvl_weight_path = os.path.join(root_path, 'model_pth/vinvl_vqa.pth')
# devlbert_weight_path = os.path.join(root_path, 'model_pth/devlbert_vqa.pth')
# uniter_weight_path = os.path.join(root_path, 'model_pth/uniter_vqa.pth')

answer_table = json.load(open('/home/datasets/mix_data/lxmert/vqa/trainval_label2ans.json'))
