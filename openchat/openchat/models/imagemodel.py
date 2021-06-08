import torch
from openchat.models.base_model import BaseModel
import openchat.config as cfg
from openchat.utils.prepare_image import detect_objects_on_single_image
from openchat.utils.transforms import build_transforms
import cv2

import sys
sys.path.insert(0, '/home/datasets/mix_data/openchat/scene_graph_benchmark-main')


class LxmertBot(BaseModel):

    def __init__(self, env, device, max_context_length):
        super().__init__('imagemodel', env)
        # self.model = MobileNetV2(num_classes=5)
        self.devices = device.lower()
        self.max_context_length = max_context_length
        # self.tokenizer = .from_pretrained()
        self.eos = '</s><s>'
        self.lxmert_model = torch.load(cfg.lxmert_weight_path)
        self.transforms = build_transforms()
        self.detect_model = torch.load(cfg.detect_weight_path)
        # self.model.to(device)

    @torch.no_grad()
    def predict(self, image_id: str, text: str) -> str:
        torch.cuda.empty_cache()
        input_ids_list: list = []
        num_of_stacked_tokens: int = 0

        print(text)

        if image_id not in self.env.histories.keys():
            self.env.clear(image_id, text)

        user_histories = reversed(self.env.histories[image_id]['user'])
        bot_histories = reversed(self.env.histories[image_id]['bot'])

        for user, bot in zip(user_histories, bot_histories):
            user_tokens = self.tokenizer.encode(user, return_tensors='pt')
            bot_tokens = self.tokenizer.encode(bot, return_tensors='pt')
            num_of_stacked_tokens += user_tokens.shape[-1] + bot_tokens.shape[-1]

            if num_of_stacked_tokens <= self.max_context_length:
                input_ids_list.append(bot_tokens)
                input_ids_list.append(user_tokens)

            else:
                break

        img_path = cfg.image_path + image_id
        img = cv2.imread(img_path)
        dets = detect_objects_on_single_image(self.detect_model, self.transforms, img)

        data = {}
        data['feats'] = torch.stack([det['features'] for det in dets]).unsqueeze(dim=0)
        data['boxes'] = torch.stack([torch.tensor(det['rect'], dtype=torch.float32) for det in dets]).unsqueeze(dim=0)

        feats = data['feats'].to('cuda')
        boxes = data['boxes'].to('cuda')
        sent = [text]

        output_dict = self.lxmert_model.model(feats, boxes, sent)

        max_score = output_dict['scores'].argmax(dim=-1)

        print(max_score)

        ans = cfg.answer_table[max_score]

        return ans
