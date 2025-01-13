import os
from datetime import datetime,timedelta 

from . import driftDetection as DD


def main(
        t,
        online_mode,
        model,
        dd_unit_int,
        cw_size,
        pw_size,
        method_code,
        threshold,
        tr_results_list,
        eval_results_list
):

    w = DD.Window()
    next_dd_date = t.start_date + timedelta(seconds=cw_size+pw_size)
    
    if online_mode:
        print("dynamic - online mode")
    else:
        print("dynamic - offline mode")
        for d_file in sorted(os.listdir(t.d_dir_path)):
            if t.end_flag: break
            f,reader = t.set_d_file(d_file)
            for row in reader:
                t.c_time = datetime.strptime(row[t.headers.index("daytime")], "%Y-%m-%d %H:%M:%S")

                if t.s_flag:
                    if t.s_filtering(): continue
                elif t.e_filtering(): break

                # --- Evaluate
                if t.c_time > t.next_eval_date:
                    with t.tf.device("/GPU:0"):
                        eval_results_list = t.call_eval(eval_results_list)
                # --- Prediction
                t.call_pred(model,row)
                # --- DD & Retraining
                
                if t.c_time > next_dd_date:
                    print(f'call_dd : {t.c_time}')
                    if DD.call_v(w.ex_cw_v(), w.ex_pw_v(), threshold):
                        tr_results_list = t.call_tr(model, w.cw, tr_results_list, t.c_time)
                    next_dd_date += timedelta(seconds=dd_unit_int)
                w.update(row[3:], t.c_time, cw_size, pw_size)
            f.close()

    return tr_results_list,eval_results_list,t.start_date,t.c_time