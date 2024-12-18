import os
from .util import driftDetection as DD, tmClass

def main(
        online_mode,
        d_dir_path,
        o_dir_path,
        start_date,
        end_date,
        model,
        epochs,
        batch_size,
        eval_unit_int,
        cw_size,
        pw_size,
        method_code,
        threshold,
        tr_results_list,
        eval_results_list
):

    t = tmClass.TerminateManager(d_dir_path, o_dir_path, start_date,
                                 end_date, eval_unit_int, epochs, batch_size)
    w = DD.Window(cw_size, pw_size, threshold, row_len=18)

    if online_mode:
        print("dynamic - online mode")
    else:
        print("dynamic - offline mode")
        for d_file in sorted(os.listdir(d_dir_path)):
            if t.end_flag: break
            f,reader = t.set_d_file(d_file)
            for row in reader:

                feature,target = t.row_converter(row)
                if t.e_filtering(t.c_time):break
                elif t.b_flag:
                    if t.b_filtering(t.c_time):continue

                # --- Evaluate
                if t.c_time > t.next_eval_date:
                    eval_results_list = t.call_eval(eval_results_list)
                # --- Prediction
                t.call_pred(model, feature=feature,target=target)
                # --- DD & Retraining
                w.update(row)
                if DD.call(method_code,w.fnum_cw(), w.fnum_pw(),w.cum_test_static):
                    tr_results_list = t.call_tr(model, w.c_window, tr_results_list)
            f.close()

    return tr_results_list,eval_results_list,t.start_date,t.c_time