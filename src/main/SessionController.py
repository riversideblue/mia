import time
from datetime import timedelta

import pandas as pd
import pytz
from sklearn.preprocessing import MinMaxScaler

from SessionDefiner import *


class SessionController:
    def __init__(self, loader):

        self.loader = loader
        self.jst = pytz.timezone("Asia/Tokyo")
        self.init_time = datetime.now(self.jst).strftime("%Y%m%d%H%M%S")
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

    def _finalize(self, current_time, session):

        self.loader.append_log('INIT_TIME', self.init_time)
        self.loader.append_log('SESSION_END_DAYTIME', current_time.isoformat())
        elapsed_time = time.time() - self.jst.localize(datetime.strptime(self.init_time, "%Y%m%d%H%M%S")).timestamp()
        self.loader.append_log('ELAPSED_TIME', elapsed_time)
        self.loader.save(f"{self.output_path}/settings_log.json")

        # --- Results processing
        add_results_col = ["nmr_fn_rate", "nmr_benign_rate"]
        add_results_list = []

        sum_fn = np.sum(session.eval_results_list[:, 5])

        # nmr_flow_num_ratio
        min_max_scaler = MinMaxScaler()
        fn_rate = session.eval_results_list[:, 5] / sum_fn
        reshaped_fn_rate = fn_rate.reshape(-1, 1)
        scaled_fn_rate = min_max_scaler.fit_transform(reshaped_fn_rate)
        add_results_list.append(scaled_fn_rate.flatten())

        # nmr_benign_ratio
        reshaped_ben_ratio = session.eval_results_list[:, 14].reshape(-1, 1)
        scaled_ben_ratio = min_max_scaler.fit_transform(reshaped_ben_ratio)
        add_results_list.append(scaled_ben_ratio.flatten())

        # Convert additional results to a 2D array
        add_results_list = np.array(add_results_list).T  # 転置して列形式に変換

        # training results
        tr_results = pd.DataFrame(session.tr_results_list, columns=self.tr_results_col)
        tr_results.to_csv(os.path.join(self.output_path, "res_train.csv"), index=False)

        # evaluate results
        eval_results = pd.DataFrame(session.eval_results_list, columns=self.eval_results_col)
        add_results = pd.DataFrame(add_results_list, columns=add_results_col)

        # Combine evaluate_results with additional_results
        eval_results = pd.concat([eval_results, add_results], axis=1)
        eval_results.to_csv(os.path.join(self.output_path, "res_eval.csv"), index=False)


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
        self._finalize(current_time, session)