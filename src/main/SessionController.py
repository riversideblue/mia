import time
from datetime import timedelta

from SessionDefiner import *


class SessionController:
    def __init__(self, loader, init_time):

        self.loader = loader
        self.init_time = init_time
        self.output_path = self._create_output_dir()
        self.tr_results_col = ["daytime", "accuracy", "loss", "training_time", "benign_count", "malicious_count", "flow_num"]
        self.tr_results_list = np.empty((0, len(self.tr_results_col)), dtype=object)
        self.eval_results_col = ["daytime", "TP", "FN", "FP", "TN", "flow_num", "TP_rate", "FN_rate", "FP_rate", "TN_rate",
                            "accuracy", "precision", "f1_score", "loss", "benign_rate"]
        self.eval_results_list = np.empty((0, len(self.eval_results_col)), dtype=object)

    def _create_output_dir(self):

        d_dir = os.path.basename(self.loader.get("DATASETS_DIR_PATH"))
        path = f"{self.loader.get('USER_DIR')}/exp/{self.init_time}_{d_dir}_{self.loader.get('RETRAINING_MODE')}_{self.loader.get('MODEL_CODE')}"
        os.makedirs(path)
        os.makedirs(f"{path}/wts")
        return path

    def _finalize(self, current_time):

        self.loader.append_log('INIT_TIME', self.init_time)
        self.loader.append_log('SESSION_END_DAYTIME', current_time.isoformat())

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

        session = session_cls(self.loader, model, self.tr_results_list, self.eval_results_list, self.output_path)
        current_time = session.run()
        self._finalize(current_time=current_time)