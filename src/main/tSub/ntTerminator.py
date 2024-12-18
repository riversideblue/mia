import os
from .util import tmClass


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
        tr_results_list,
        eval_results_list
):

    t = tmClass.TerminateManager(d_dir_path,o_dir_path,start_date,
                                 end_date,eval_unit_int,epochs,batch_size)

    if online_mode:
        print("- < static/online mode activate >")
    else:
        print("- < static/offline mode activate >")
        for d_file in sorted(os.listdir(d_dir_path)):
            if t.end_flag: break
            f,reader = t.set_d_file(d_file)
            for row in reader:

                feature,target = t.row_converter(row)
                if t.e_filtering(t.c_time):
                    break
                elif t.b_flag:
                    if t.b_filtering(t.c_time): continue

                # --- Evaluate
                if t.c_time > t.next_eval_date:
                    eval_results_list = t.call_eval(eval_results_list)
                # --- Prediction
                t.call_pred(model, feature=feature, target=target)
            f.close()

    return tr_results_list,eval_results_list,t.start_date,t.c_time