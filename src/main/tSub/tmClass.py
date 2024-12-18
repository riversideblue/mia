import csv
from datetime import timedelta, datetime

import numpy as np
import pandas as pd

from .util import modelEvaluator,modelTrainer


class TerminateManager:
    def __init__(self,d_dir_path,o_dir_path,start_date,end_date,rtr_int,eval_unit_int,epochs,batch_size):

        self.y_true = []
        self.y_pred = []
        self.headers = []
        self.eval_list = []
        self.b_flag = True
        self.first_row_flag = True
        self.end_flag = False

        self.d_dir_path = d_dir_path
        self.o_dir_path = o_dir_path
        self.c_time = start_date
        self.start_date = start_date
        self.end_date = end_date
        self.rtr_int = rtr_int
        self.next_rtr_date = start_date + timedelta(seconds=rtr_int)
        self.eval_unit_int = eval_unit_int
        self.next_eval_date = start_date + timedelta(seconds=eval_unit_int)
        self.epochs = epochs
        self.batch_size = batch_size

    def set_d_file(self, d_file):
        d_file_path: str = f"{self.d_dir_path}/{d_file}"
        print(f"- {d_file} set now")
        f = open(d_file_path, mode='r')
        reader = csv.reader(f)
        self.headers = next(reader)
        return f,reader

    def row_converter(self,row):
        self.c_time = datetime.strptime(row[self.headers.index("daytime")], "%Y-%m-%d %H:%M:%S")
        feature = np.array(row[3:-1], dtype=np.float32)
        target = int(row[self.headers.index("label")])
        return feature, target

    # 最初の行の時刻が開始時刻より前 => 開始時刻まで何もしない return True
    # 最初の行ではない行が開始時刻より前 => 開始時刻まで何もしない return True
    # 最初の行ではない行が開始時刻より後 => 処理を開始 return False
    # 最初の行の時刻が開始時刻より後 => 最初の行の時刻を開始時刻に合わせて処理を開始 return False

    def b_filtering(self, c_time): # return False then start processing
        if self.first_row_flag:
            self.first_row_flag = False
            if c_time > self.start_date:
                delta = c_time - self.start_date
                self.start_date = c_time
                self.end_date += delta
                self.next_rtr_date += delta
                self.next_eval_date += delta
                self.b_flag = False
                return False
            elif c_time == self.start_date:
                self.b_flag = False
                return False
            elif c_time < self.start_date:
                self.b_flag = True
                return True
        elif c_time < self.start_date:
            return True
        else:
            self.b_flag = False
            return False

    def e_filtering(self, c_time):# if False keep Processing
        if c_time > self.end_date:
            print("- < detected end_daytime >")
            self.end_flag = True
            return True
        return False

    def call_eval(self,list_eval_results):
        print("--- evaluate model")
        evaluate_daytime = self.next_eval_date - timedelta(seconds=self.eval_unit_int / 2)
        eval_arr = modelEvaluator.main(self.y_true, self.y_pred)
        eval_arr = np.append([evaluate_daytime], eval_arr)
        list_eval_results = np.vstack([list_eval_results, eval_arr])
        self.y_true = []
        self.y_pred = []
        self.next_eval_date += timedelta(seconds=self.eval_unit_int)
        return list_eval_results

    def call_pred(self,model,feature,target):
        self.y_pred.append(model.predict_on_batch(feature.reshape(1, -1))[0][0])
        self.y_true.append(target)

    def call_tr(self, model, rtr_list, rtr_results_list):
        df = pd.DataFrame(np.array(rtr_list))
        features = df.iloc[:, 3:-1].astype(float)
        targets = df.iloc[:, -1].astype(int)
        retraining_daytime = datetime.strptime(df.iloc[-1, 2], "%Y-%m-%d %H:%M:%S")
        model, arr_rtr_results = modelTrainer.main(
            model=model,
            features=features,
            targets=targets,
            output_dir_path=self.o_dir_path,
            epochs=self.epochs,
            batch_size=self.batch_size,
            retraining_daytime=retraining_daytime
        )
        self.next_rtr_date += timedelta(seconds=self.rtr_int)
        rtr_results_list = np.vstack([rtr_results_list, arr_rtr_results])
        return rtr_results_list