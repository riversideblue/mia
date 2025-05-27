import csv
import os
import tensorflow as tf

from Evaluator import evaluate
from Trainer import train
from DriftDetection import *

class NoRetrainSession:
    def __init__(self, loader, model, tr_results_list, eval_results_list, output_path):

        self.y_trues = []
        self.y_preds = []
        self.loader = loader
        self.model = model
        self.tr_results_list = tr_results_list
        self.eval_results_list = eval_results_list
        self.output_path = output_path
        self.session_start_flag = True
        self.first_row_flag = True
        self.session_end_flag = False
        self.session_start_date = datetime.strptime(loader.get('SESSION_START_DATE'), "%Y-%m-%d %H:%M:%S")
        self.current_time = self.session_start_date
        self.rtr_int = loader.get('RETRAINING_INTERVAL')
        self.next_rtr_date = self.session_start_date + timedelta(seconds=self.rtr_int)
        self.eval_unit_int = loader.get('EVALUATE_UNIT_INTERVAL')
        self.next_eval_date = self.session_start_date + timedelta(seconds=self.eval_unit_int)
        self.epochs = loader.get('TrainingDefine')['EPOCHS']
        self.batch_size = loader.get('TrainingDefine')['BATCH_SIZE']
        target_range = timedelta(
            days=loader.get("TargetRange")["DAYS"],
            hours=loader.get("TargetRange")["HOURS"],
            minutes=loader.get("TargetRange")["MINUTES"],
            seconds=loader.get("TargetRange")["SECONDS"]
        )
        self.session_end_date = self.session_start_date + target_range
        self.row_headers = []

    def run(self):
        for d_file in sorted(os.listdir(self.loader.get('DATASETS_DIR_PATH'))):
            if self.session_end_flag: break
            f, reader = self._set_d_file(d_file)
            for row in reader:
                if self._handle_row(row):
                    return self.current_time
            f.close()
        return self.current_time

    def _set_d_file(self, d_file):
        d_file_path: str = f"{self.loader.get('DATASETS_DIR_PATH')}/{d_file}"
        print(f"- {d_file} set now")
        f = open(d_file_path, mode='r')
        reader = csv.reader(f)
        self.row_headers = next(reader)
        return f,reader

    def _handle_row(self, row):
        self.current_time = datetime.strptime(row[self.row_headers.index("daytime")], "%Y-%m-%d %H:%M:%S")
        if self.session_start_flag and self._start_filtering():
            return False
        if self._end_filtering(): return True
        self._evaluate_if_needed()
        self.y_preds.append(self._predict(self.model, row))
        self.y_trues.append(int(row[-1]))
        self._retrain_if_needed(self.model, row)
        return False

    def _start_filtering(self): # return False then start processing
        if self.first_row_flag:
            self.first_row_flag=False
            if self.current_time > self.session_start_date:
                delta=self.current_time-self.session_start_date
                self.session_start_date=self.current_time
                self.session_end_date+=delta
                self.next_rtr_date+=delta
                self.next_eval_date+=delta
                self.s_flag=False
                return False
            elif self.current_time == self.session_start_date:
                self.s_flag = False
                return False
            elif self.current_time < self.session_start_date:
                self.s_flag = True
                return True
            else: pass
        elif self.current_time < self.session_start_date:
            return True
        else:
            self.s_flag = False
            return False

    def _end_filtering(self):# if False keep Processing
        if self.current_time > self.session_end_date:
            print("- < detected end_daytime >")
            self.end_flag = True
            return True
        return False

    def _evaluate_if_needed(self):
        if self.current_time > self.next_eval_date:
            with tf.device("/GPU:0"):
                print("--- evaluate model")
                evaluate_daytime = self.next_eval_date - timedelta(seconds=self.eval_unit_int / 2)
                eval_arr = [evaluate_daytime] + evaluate(self.y_trues, self.y_preds, tf)
                self.eval_results_list = np.vstack([self.eval_results_list, eval_arr])
                self.y_trues.clear()
                self.y_preds.clear()
                self.next_eval_date += timedelta(seconds=self.eval_unit_int)

    def _predict(self, model, row):
        tensor_input = tf.convert_to_tensor([list(map(float, row[3:-1]))], dtype=tf.float32)
        prediction_result = model(tensor_input,training=False)[0][0].numpy()
        return prediction_result

    def _retrain_if_needed(self, model, row):
        pass

class DynamicSession(NoRetrainSession):
    def __init__(self, loader, model, tr_results_list, eval_results_list, output_path):
        super().__init__(loader, model, tr_results_list, eval_results_list, output_path)
        self.dd_settings = loader.get('DriftDetection')
        self.wm = WindowManager(model, self.dd_settings.get('WindowConfig'))
        self.y_preds_by_window = [[] for _ in self.wm.windows]
        # PW+CWの最大値 = 18000
        self.next_dd_date = self.session_start_date + timedelta(seconds=18000)

    def _handle_row(self, row):
        self.current_time = datetime.strptime(row[self.row_headers.index("daytime")], "%Y-%m-%d %H:%M:%S")
        if self.session_start_flag and self._start_filtering(): return
        if self._end_filtering(): return

        self._evaluate_if_needed()
        self.y_trues.append(int(row[-1]))
        for i, w in enumerate(self.wm.windows):
            self.y_preds_by_window[i].append(self._predict(w.model, row))
            self._retrain_if_needed(w.model, row)
        last_column = [y_pred_by_window[-1] for y_pred_by_window in self.y_preds_by_window]
        self.y_preds.append((sum(last_column) > len(self.y_preds_by_window)) // 2)

    def _retrain_if_needed(self, model, row):
        if self.current_time > self.next_dd_date:
            if self.wm.detect_by_a_majority(self.dd_settings.get('METHOD_CODE'),self.dd_settings.get('K')):
                for w in self.wm.windows:
                    self.tr_results_list = train(model, w.cw, self.y_trues, self.output_path, self.epochs, self.batch_size,self.current_time)
            self.next_dd_date += timedelta(seconds=self.dd_settings.get('DRIFT_DETECTION_UNIT_INTERVAL'))
        self.wm.update_all(row[3:], self.current_time)

class StaticSession(NoRetrainSession):
    def __init__(self, loader, model, tr_results_list, eval_results_list, output_path):
        super().__init__(loader, model, tr_results_list, eval_results_list, output_path)
        self.rtr_list = []

    def _retrain_if_needed(self, model, row):
        if self.current_time > self.next_rtr_date:
            with tf.device("/GPU:0"):
                self.tr_results_list = train(model, self.rtr_list, self.y_trues, self.output_path, self.epochs, self.batch_size, self.current_time)
                self.rtr_list = []
        self.rtr_list.append(np.array(row[3:], dtype=float))
