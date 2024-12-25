import os
from datetime import datetime

def main(
        t,
        online_mode,
        model,
        tr_results_list,
        eval_results_list
):

    if online_mode:
        print("- < static/online mode activate >")
    else:
        print("- < static/offline mode activate >")
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
            f.close()

    return tr_results_list,eval_results_list,t.start_date,t.c_time