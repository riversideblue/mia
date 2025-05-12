import csv
import os
from datetime import datetime, timedelta

import numpy as np


class NoRetrainSession:
    def __init__(self, loader, model, tr_results, eval_results):

        self.loader = loader
        self.model = model
        self.tr_results = tr_results
        self.eval_results = eval_results
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
        return self.tr_results, self.eval_results, self.current_time

    def _set_d_file(self, d_file):
        d_file_path: str = f"{self.loader.get('DATASETS_DIR_PATH')}/{d_file}"
        print(f"- {d_file} set now")
        f = open(d_file_path, mode='r')
        reader = csv.reader(f)
        self.row_headers = next(reader)
        return f,reader

    def _handle_row(self, row):
        self.current_time = datetime.strptime(row[self.row_headers.index("daytime")], "%Y-%m-%d %H:%M:%S")
        if self.s_flag and self._start_filtering(): return
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
            with self.tf.device("/GPU:0"):
                print("--- evaluate model")
                evaluate_daytime = self.next_eval_date - timedelta(seconds=self.eval_unit_int / 2)
                eval_arr = [evaluate_daytime] + modelEvaluator.main(self.y_true, self.y_pred, self.tf)
                eval_res_li = np.vstack([eval_res_li, eval_arr])
                self.y_true.clear()
                self.y_pred.clear()
                self.next_eval_date += timedelta(seconds=self.eval_unit_int)

    def _predict(self, row):
        self.t.call_pred(self.model, row)

    def _retrain_if_needed(self, row):
        raise NotImplementedError

class DynamicSession(NoRetrainSession):
    def __init__(self, t, model, tr_results, eval_results, dd_params):
        super().__init__(t, model, tr_results, eval_results)
        self.dd_unit_int, self.cw_size, self.pw_size, self.method_code, self.k, self.threshold, self.obs_mode = dd_params
        self.w = DD.Window()
        self.next_dd_date = self.t.start_date + timedelta(seconds=self.cw_size + self.pw_size)

    def _retrain_if_needed(self, row):
        if self.t.c_time > self.next_dd_date:
            if DD.call(self.method_code, self.w.ex_cw_v(), self.w.ex_pw_v(), self.threshold, self.k):
                self.tr_results = self.t.call_tr(self.model, self.w.cw, self.tr_results, self.t.c_time)
            self.next_dd_date += timedelta(seconds=self.dd_unit_int)
        self.w.update(row[3:], self.t.c_time, self.cw_size, self.pw_size)

class StaticSession(NoRetrainSession):
    def __init__(self, t, model, tr_results, eval_results):
        super().__init__(t, model, tr_results, eval_results)
        self.rtr_list = []

    def _retrain_if_needed(self, row):
        if self.t.c_time > self.t.next_rtr_date:
            with self.t.tf.device("/GPU:0"):
                self.tr_results = self.t.call_tr(self.model, self.rtr_list, self.tr_results, self.t.c_time)
                self.rtr_list = []
        self.rtr_list.append(np.array(row[3:], dtype=float))
