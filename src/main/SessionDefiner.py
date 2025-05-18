import csv
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import tensorflow as tf

from Evaluator import evaluate
from Trainer import train
from DriftDetection import *

class NoRetrainSession:
    def __init__(self, loader, model):

        self.y_true = []
        self.y_pred = []
        self.loader = loader
        self.model = model
        self.tr_results_list = []
        self.eval_results_list = []
        self.session_start_flag = True
        self.first_row_flag = True
        self.session_end_flag = False
        self.session_start_date = loader.get('SESSION_START_DATE')
        self.current_time = self.session_start_date
        self.rtr_int = loader.get('RTR_INT')
        self.next_rtr_date = self.session_start_date + timedelta(seconds=self.rtr_int)
        self.eval_unit_int = loader.get('EVAL_UNIT_INT')
        self.next_eval_date = self.session_start_date + timedelta(seconds=self.eval_unit_int)
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
                self._handle_row(row)
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
        if self.session_start_flag and self._start_filtering(): return
        if self._end_filtering(): return
        self._evaluate_if_needed()
        self._predict(row)
        self._retrain_if_needed(row)

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
                eval_arr = [evaluate_daytime] + evaluate(self.y_true, self.y_pred, tf)
                self.eval_results_list = np.vstack([self.eval_results_list, eval_arr])
                self.y_true.clear()
                self.y_pred.clear()
                self.next_eval_date += timedelta(seconds=self.eval_unit_int)

    def _predict(self, row):
        tensor_input = tf.convert_to_tensor([list(map(float, row[3:-1]))], dtype=tf.float32)
        self.y_pred.append(self.model(tensor_input,training=False)[0][0].numpy())
        self.y_true.append(int(row[-1]))

    def _retrain_if_needed(self, row):
        raise NotImplementedError

class DynamicSession(NoRetrainSession):
    def __init__(self, loader, model):
        super().__init__(loader, model)
        self.dd_settings = loader.get('DriftDetection')
        self.w = DetectionWindow()
        self.next_dd_date = self.session_start_date + timedelta(seconds=self.dd_settings.get('CW_SIZE') + self.dd_settings.get('PW_SIZE'))

    def _retrain_if_needed(self, row):
        if self.current_time > self.next_dd_date:
            if call(self.loader.get('METHOD_CODE'), self.w.ex_cw_v(), self.w.ex_pw_v(), self.loader.get('THRESHOLD'), self.loader.get('K')):
                self.tr_results_list = train(self.model, self.w.cw, self.tr_results_list, self.current_time)
            self.next_dd_date += timedelta(seconds=self.dd_settings.get('DD_UNIT_INT'))
        self.w.update(row[3:], self.current_time)

class StaticSession(NoRetrainSession):
    def __init__(self, loader, model):
        super().__init__(loader, model)
        self.rtr_list = []

    def _retrain_if_needed(self, row):
        if self.current_time > self.next_rtr_date:
            with tf.device("/GPU:0"):
                self.tr_results_list = train(self.model, self.rtr_list, self.tr_results_list, self.current_time)
                self.rtr_list = []
        self.rtr_list.append(np.array(row[3:], dtype=float))
