import os
from datetime import datetime

from . import driftDetection as DD


def main(
        t,
        online_mode,
        model,
        cw_size,
        pw_size,
        method_code,
        threshold,
        tr_results_list,
        eval_results_list
):

    w = DD.Window()

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
                w.update(row[3:], t.c_time, cw_size, pw_size)
                w.cum_statics+=DD.call(method_code, w.v2_cw(), w.v2_pw())
                print(f"cum_statics: {w.cum_statics}")
                while abs(w.cum_statics) > threshold:
                    tr_results_list = t.call_tr(model, w.cw, tr_results_list, t.c_time)
                    w.cum_statics -= threshold if w.cum_statics >= 0 else -threshold
            f.close()

    return tr_results_list,eval_results_list,t.start_date,t.c_time