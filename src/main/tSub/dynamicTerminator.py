import os
from .util import driftDetection as DD

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
    w = DD.Window(cw_size, pw_size, row_len=15)
    counter = 0
    if online_mode:
        print("dynamic - online mode")
    else:
        print("dynamic - offline mode")
        for d_file in sorted(os.listdir(t.d_dir_path)):
            if t.end_flag: break
            f,reader = t.set_d_file(d_file)
            for row in reader:

                feature,target = t.row_converter(row)
                if t.s_flag:
                    if t.s_filtering(t.c_time): continue
                elif t.e_filtering(t.c_time): break

                conter += 1
                print(counter)

                # --- Evaluate
                if t.c_time > t.next_eval_date:
                    eval_results_list = t.call_eval(eval_results_list)
                # --- Prediction
                t.call_pred(model, feature=feature,target=target)
                # --- DD & Retraining
                w.update(row[3:])
                print(f"cum:{w.cum_test_static}")
                w.cum_test_static += DD.call(method_code, w.v2_cw(), w.v2_pw())
                slope = w.cum_test_static/threshold
                print(slope)
                if slope > 1:
                    t.epochs = int(slope)
                    tr_results_list = t.call_tr(model, w.c_window, tr_results_list)
                    w.cum_test_static = w.cum_test_static%threshold

            f.close()

    return tr_results_list,eval_results_list,t.start_date,t.c_time