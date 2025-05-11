import os
from datetime import datetime


class BaseSession:
    def __init__(self, t, model, tr_results, eval_results):
        self.t = t
        self.model = model
        self.tr_results = tr_results
        self.eval_results = eval_results

    def run(self):
        for d_file in sorted(os.listdir(self.t.d_dir_path)):
            if self.t.end_flag: break
            f, reader = self.t.set_d_file(d_file)
            for row in reader:
                self._handle_row(row)
            f.close()
        return self.tr_results, self.eval_results, self.t.start_date, self.t.c_time

    def _handle_row(self, row):
        self.t.c_time = datetime.strptime(row[self.t.headers.index("daytime")], "%Y-%m-%d %H:%M:%S")
        if self.t.s_flag and self.t.s_filtering(): return
        if self.t.e_filtering(): return
        self._evaluate_if_needed()
        self._predict(row)
        self._retrain_if_needed(row)

    def _evaluate_if_needed(self):
        if self.t.c_time > self.t.next_eval_date:
            with self.t.tf.device("/GPU:0"):
                self.eval_results = self.t.call_eval(self.eval_results)

    def _predict(self, row):
        self.t.call_pred(self.model, row)

    def _retrain_if_needed(self, row):
        raise NotImplementedError

class DynamicSession(BaseSession):
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

class StaticSession(BaseSession):
    def __init__(self, t, model, tr_results, eval_results):
        super().__init__(t, model, tr_results, eval_results)
        self.rtr_list = []

    def _retrain_if_needed(self, row):
        if self.t.c_time > self.t.next_rtr_date:
            with self.t.tf.device("/GPU:0"):
                self.tr_results = self.t.call_tr(self.model, self.rtr_list, self.tr_results, self.t.c_time)
                self.rtr_list = []
        self.rtr_list.append(np.array(row[3:], dtype=float))

class NoRetrainSession(BaseSession):
    def __init__(self, t, model, tr_results, eval_results):
        super().__init__(t, model, tr_results, eval_results)
        self.rtr_list = []