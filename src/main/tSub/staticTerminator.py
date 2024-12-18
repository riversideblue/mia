import os
from datetime import timedelta

def main(
        t,
        online_mode,
        model,
        rtr_int,
        tr_results_list,
        eval_results_list
):

    rtr_list = []

    if online_mode:
        print("- < static/online mode activate >")
    else:
        print("- < static/offline mode activate >")
        for d_file in sorted(os.listdir(t.d_dir_path)):
            if t.end_flag: break
            f,reader = t.set_d_file(d_file)
            for row in reader:

                feature,target = t.row_converter(row)
                if t.s_flag:
                    if t.s_filtering(t.c_time): continue
                elif t.e_filtering(t.c_time): break

                # --- Evaluate
                if t.c_time > t.next_eval_date:
                    eval_results_list = t.call_eval(eval_results_list)
                # --- Prediction
                t.call_pred(model, feature=feature, target=target)
                # --- Retraining
                if t.c_time > t.next_rtr_date:
                    tr_results_list = t.call_tr(model, rtr_list, tr_results_list)
                rtr_list.append(row)
            f.close()

    return tr_results_list,eval_results_list,t.start_date,t.c_time