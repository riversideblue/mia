import json
import os
from datetime import datetime
import pytz

class SettingsLoader:

    def __init__(self, path="src/main/settings.json"):
        with open(path, "r") as f:
            self.settings = json.load(f)
        print(json.dumps(self.settings, indent=2, ensure_ascii=False))
        self._init_log()
        self._configure_environment()

    def _init_log(self):
        jst = pytz.timezone("Asia/Tokyo")
        init_time = datetime.now(jst).strftime("%Y%m%d%H%M%S")
        self.settings["Log"] = {"INIT_TIME": init_time}

    def _configure_environment(self):
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = self.settings["OS"]["TF_CPP_MIN_LOG_LEVEL"]
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = self.settings["OS"]["TF_FORCE_GPU_ALLOW_GROWTH"]
        os.environ["CUDA_VISIBLE_DEVICES"] = self.settings["OS"]["CUDA_VISIBLE_DEVICES"]

    def get(self, key):
        return self.settings[key]

    def append_log(self, key, value):
        if "Log" not in self.settings:
            self.settings["Log"] = {}
        self.settings["Log"][key] = value

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.settings, f, indent=1)
