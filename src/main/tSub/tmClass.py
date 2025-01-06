import csv
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from . import modelEvaluator, modelTrainer

class TerminateManager:
    def __init__(self,tf,d_dir_path,o_dir_path,start_date,end_date,rtr_int,eval_unit_int,epochs,batch_size):

        self.tf = tf
        self.y_true = []
        self.y_pred = []
        self.headers = []
        self.eval_list = []
        self.s_flag = True
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
        self.scaler = StandardScaler()
        self.scaled_flag = False

    def set_d_file(self, d_file):
        d_file_path: str = f"{self.d_dir_path}/{d_file}"
        print(f"- {d_file} set now")
        f = open(d_file_path, mode='r')
        reader = csv.reader(f)
        self.headers = next(reader)
        return f,reader

    def s_filtering(self): # return False then start processing
        if self.first_row_flag:
            self.first_row_flag=False
            if self.c_time > self.start_date:
                delta=self.c_time-self.start_date
                self.start_date=self.c_time
                self.end_date+=delta
                self.next_rtr_date+=delta
                self.next_eval_date+=delta
                self.s_flag=False
                return False
            elif self.c_time == self.start_date:
                self.s_flag = False
                return False
            elif self.c_time < self.start_date:
                self.s_flag = True
                return True
            else: pass
        elif self.c_time < self.start_date:
            return True
        else:
            self.s_flag = False
            return False

    def e_filtering(self):# if False keep Processing
        if self.c_time > self.end_date:
            print("- < detected end_daytime >")
            self.end_flag = True
            return True
        return False

    def call_eval(self, eval_res_li):
        print("--- evaluate model")
        evaluate_daytime = self.next_eval_date - timedelta(seconds=self.eval_unit_int / 2)
        eval_arr = [evaluate_daytime] + modelEvaluator.main(self.y_true, self.y_pred, self.tf)
        eval_res_li = np.vstack([eval_res_li, eval_arr])
        self.y_true.clear()
        self.y_pred.clear()
        self.next_eval_date += timedelta(seconds=self.eval_unit_int)
        return eval_res_li

    def call_pred(self,model,row):
        tensor_input = self.tf.convert_to_tensor([list(map(float, row[3:-1]))], dtype=self.tf.float32)
        self.y_pred.append(model(tensor_input,training=False)[0][0].numpy())
        self.y_true.append(int(row[-1]))

    def call_tr(self, model, rtr_list, rtr_res_li, c_time):
        df = pd.DataFrame(rtr_list).dropna()
        features = df.iloc[:, :-1]
        targets = df.iloc[:, -1]
        # scaled_features = self.scaler.fit_transform(features)
        model, rtr_res_arr = modelTrainer.main(
            model=model,
            features=features,
            targets=targets,
            output_dir_path=self.o_dir_path,
            epochs=self.epochs,
            batch_size=self.batch_size,
            rtr_date=c_time
        )
        self.next_rtr_date += timedelta(seconds=self.rtr_int)
        rtr_res_li = np.vstack([rtr_res_li, rtr_res_arr])
        return rtr_res_li