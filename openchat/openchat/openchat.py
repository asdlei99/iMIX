from typing import Union
import os
from openchat.envs import BaseEnv, TerminalEnv
import torch
import json
from .models.imagemodel import LxmertBot
import openchat.config as cfg

class OpenChat(object):

    def __init__(
        self,
        model: str,
        device: str = "cpu",
        env: Union[BaseEnv, str] = TerminalEnv(),
        max_context_length=128,
    ) -> None:
        """
        Constructor for OpenChat

        Args:
            env (Union[BaseEnv, str]): dialogue environment
            model (str): generative dialogue model
            size (str): model size (It may vary depending on the model)
            device (str): device argument
            max_context_length (int): max history context length
                (it means that length of input context tokens)
        """

        print("""
           ____   ____   ______ _   __   ______ __  __ ___   ______
          / __ \ / __ \ / ____// | / /  / ____// / / //   | /_  __/
         / / / // /_/ // __/  /  |/ /  / /    / /_/ // /| |  / /   
        / /_/ // ____// /___ / /|  /  / /___ / __  // ___ | / /    
        \____//_/    /_____//_/ |_/   \____//_/ /_//_/  |_|/_/     
                        
                             ... LOADING ...
        """)

        self.device = device
        self.max_context_length = max_context_length
        self.env = env

        self.model = self.select_model(model)
        self.model.run()

    def select_model(self, model):
        assert model in self.available_models(), \
            f"Unsupported model. available models: {self.available_models()}"

        if model == "vqa_model":

            device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
            print(device)

            model = LxmertBot(env=self.env,
                             max_context_length=self.max_context_length,
                             device=self.device, )
            return model


    def available_models(self):
        return ["dialogpt", "vqa_model"]
