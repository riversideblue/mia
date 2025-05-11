import time
from datetime import timedelta

from SessionDefiner import *


class SessionController:
    def __init__(self, loader, init_time):

        self.loader = loader
        self.init_time = init_time
        target_range = timedelta(
            days=loader.get("TargetRange")["DAYS"],
            hours=loader.get("TargetRange")["HOURS"],
            minutes=loader.get("TargetRange")["MINUTES"],
            seconds=loader.get("TargetRange")["SECONDS"]
        )
        self.session_end_date = loader.get('SESSION_START_DATE') + target_range
        self.output_path = self._create_output_dir()
        self.tr_results_list = []
        self.eval_results_list = []

    def _create_output_dir(self):

        d_dir = os.path.basename(self.loader.get("DATASETS_DIR_PATH"))
        path = f"{self.loader.get('USER_DIR')}/exp/{self.init_time}_{d_dir}_{self.loader.get('RETRAINING_MODE')}_{self.loader('MODEL_CODE')}"
        os.makedirs(path)
        os.makedirs(f"{path}/wts")
        return path

    def _finalize(self, current_time):

        self.loader.append_log('INIT_TIME', self.init_time)
        self.loader.append_log('START_DAYTIME', self.loader.get('SESSION_START_DATE').isoformat())
        self.loader.append_log('END_DAYTIME', current_time.isoformat())

        elapsed_time = time.time() - self.init_time
        self.loader.append_log('ELAPSED_TIME', elapsed_time)

        self.loader.save(f"{self.output_path}/settings_log.json")


    def run(self, model):

        mode_map = {
            "dy": DynamicSession,
            "st": StaticSession,
            "nt": NoRetrainSession
        }
        mode = self.loader.get("RETRAINING_MODE")
        session_cls = mode_map.get(mode)
        if not session_cls:
            raise ValueError(f"Invalid RETRAINING_MODE: {mode}")

        session = session_cls(self.settings, model)
        self._finalize(current_time=time.time())
        self.tr_results_list, self.eval_results_list = session.run()