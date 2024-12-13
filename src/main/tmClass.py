import sys
from datetime import timedelta, datetime

import numpy as np
import pandas as pd

from main import modelEvaluator, modelTrainer


class TerminateManager:
    def __init__(self,beginning_dtime,end_dtime,eval_unit_int,o_dir_path,epochs,batch_size):
        self.y_true = []
        self.y_pred = []
        self.c_time = beginning_dtime
        self.beginning_dtime = beginning_dtime
        self.b_flag = True
        self.first_row_flag = True
        self.end_flag = False
        self.end_dtime = end_dtime
        self.eval_unit_int = eval_unit_int
        self.next_eval_dtime = beginning_dtime + timedelta(seconds=self.eval_unit_int)
        self.o_dir_path = o_dir_path
        self.epochs = epochs
        self.batch_size = batch_size

    def b_filtering(self, c_time):
        if self.first_row_flag:
            if c_time > self.beginning_dtime:
                print("- error : beginning_daytime should be within datasets range")
                sys.exit(1)
            else:
                self.first_row_flag = False
        elif not c_time < self.beginning_dtime:
            self.b_flag = False
            return False
        return True

    def e_filtering(self, c_time):
        if c_time > self.end_dtime:  # timestampがend_daytimeを超えた時
            print("- < detected end_daytime >")
            self.end_flag = True
            return True
        return False

    def call_eval(self,list_eval_results):
        print("--- evaluate model")
        evaluate_daytime = self.next_eval_dtime - timedelta(seconds=self.eval_unit_int / 2)
        evaluate_results_array = modelEvaluator.main(self.y_true, self.y_pred)
        evaluate_results_array = np.append([evaluate_daytime], evaluate_results_array)
        list_eval_results = np.vstack([list_eval_results, evaluate_results_array])
        self.y_true = []
        self.y_pred = []
        self.next_eval_dtime += timedelta(seconds=self.eval_unit_int)
        return list_eval_results

    def call_pred(self,model,feature,target):
        self.y_pred.append(model.predict_on_batch(feature.reshape(1, -1))[0][0])
        self.y_true.append(target)

    def call_rtr(self, model, rtr_list, rtr_results_list):
        print("--- Drift Detected")
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
        rtr_results_list = np.vstack([rtr_results_list, arr_rtr_results])
        return rtr_results_list