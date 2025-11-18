import json
import os


class Config:
    def __init__(self, config_dir='config'):
        self.config_dir = config_dir
        self.model_config = self.load_config('model_config.json')
        self.paths_config = self.load_config('paths.json')

    def load_config(self, filename):
        config_path = os.path.join(self.config_dir, filename)
        with open(config_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    def get_model_config(self):
        return self.model_config

    def get_paths_config(self):
        return self.paths_config
