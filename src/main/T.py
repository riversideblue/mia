import json
import pickle
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import pytz

from sklearn.preprocessing import MinMaxScaler
from tSub import *

def is_pass_exist(path):
    if not os.path.exists(path):  # error >> argument 1 name
        print(f"Cannot find training dataset: {os.path.basename(path)}")
        sys.exit(1)

# ----- Main

def main():

    # --- Load settings
    settings = json.load(open("src/main/settingsT.json", "r"))
    settings["Log"] = {}

    # --- Get current time in JST
    jst = pytz.timezone("Asia/Tokyo")
    init_time: str = datetime.now(jst).strftime("%Y%m%d%H%M%S")
    settings["Log"]["INIT_TIME"] = init_time

    # --- Setting
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = settings["OS"]["TF_CPP_MIN_LOG_LEVEL"]  # log amount
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = settings["OS"]["TF_FORCE_GPU_ALLOW_GROWTH"]  # gpu mem limit
    os.environ["CUDA_VISIBLE_DEVICES"] = settings["OS"]["CUDA_VISIBLE_DEVICES"]  # cpu : -1

    # --- Field
    d_dir_path: str = settings["DATASETS_DIR_PATH"]
    start_date = datetime.strptime(settings["START_DATE"], "%Y-%m-%d %H:%M:%S")
    days: int = settings["TargetRange"]["DAYS"]
    hours: int = settings["TargetRange"]["HOURS"]
    minutes: int = settings["TargetRange"]["MINUTES"]
    seconds: int = settings["TargetRange"]["SECONDS"]
    target_range = timedelta(
        days=days,
        hours=hours,
        minutes=minutes,
        seconds=seconds
    )
    end_date: datetime = start_date + target_range
    epochs: int = settings["TrainingDefine"]["EPOCHS"]
    batch_size: int = settings["TrainingDefine"]["BATCH_SIZE"]

    online_mode: bool = bool(settings["ONLINE_MODE"])
    rtr_mode = settings["RETRAINING_MODE"]
    rtr_int: int = settings["RETRAINING_INTERVAL"]
    eval_unit_int: int = settings["EVALUATE_UNIT_INTERVAL"]

    cw_size = settings["DriftDetection"]["CURRENT_WIN_SIZE"]
    pw_size = settings["DriftDetection"]["PAST_WIN_SIZE"]
    method_code = settings["DriftDetection"]["METHOD_CODE"]
    threshold = settings["DriftDetection"]["THRESHOLD"]

    is_pass_exist(d_dir_path)

    # --- Create output directory
    o_dir_path: str = f"exp/{init_time}"
    os.makedirs(o_dir_path)
    os.makedirs(f"{o_dir_path}/model_weights")

    # --- Set results
    t_results_col = ["daytime", "accuracy", "loss", "training_time", "benign_count", "malicious_count", "flow_num"]
    tr_results_list = np.empty((0,len(t_results_col)),dtype=object)
    eval_results_col = ["daytime", "TP", "FN", "FP", "TN", "flow_num", "TP_rate", "FN_rate", "FP_rate", "TN_rate", "accuracy", "precision", "f1_score", "loss", "benign_rate"]
    eval_results_list = np.empty((0,len(eval_results_col)),dtype=object)

    # --- Foundation model setting
    f_model_path = settings["FOUNDATION_MODEL_PATH"]
    model = modelCreator.main()
    if f_model_path == "":
        print("- start with new model ...")
    elif os.path.exists(f_model_path):
        with open(f_model_path, 'rb') as f:
            print(f"- start with exist model {f_model_path} ...")
            init_weights = pickle.load(f)
            model.set_weights(init_weights)
    else:
        print("- invalid path")
        sys.exit(1)

    process_start_date = time.time()

    # --- Terminate
    if rtr_mode == "dynamic":
         tr_results_list,eval_results_list,start_date,end_date = dynamicTerminator.main(
             online_mode= online_mode,
             d_dir_path=d_dir_path,
             o_dir_path=o_dir_path,
             start_date=start_date,
             end_date=end_date,
             model=model,
             epochs=epochs,
             batch_size=batch_size,
             eval_unit_int=eval_unit_int,
             cw_size=cw_size,
             pw_size=pw_size,
             method_code=method_code,
             threshold=threshold,
             tr_results_list=tr_results_list,
             eval_results_list=eval_results_list
         )
    elif rtr_mode == "static":
        tr_results_list,eval_results_list,start_date,end_date = staticTerminator.main(
            online_mode=online_mode,
            d_dir_path=d_dir_path,
            o_dir_path=o_dir_path,
            start_date=start_date,
            end_date=end_date,
            model=model,
            epochs=epochs,
            batch_size=batch_size,
            eval_unit_int=eval_unit_int,
            rtr_int=rtr_int,
            tr_results_list=tr_results_list,
            eval_results_list=eval_results_list
        )
    elif rtr_mode == "non-training":
        tr_results_list,eval_results_list,start_date,end_date = ntTerminator.main(
            online_mode=online_mode,
            d_dir_path=d_dir_path,
            o_dir_path=o_dir_path,
            start_date=start_date,
            end_date=end_date,
            model=model,
            epochs=epochs,
            batch_size=batch_size,
            eval_unit_int=eval_unit_int,
            tr_results_list=tr_results_list,
            eval_results_list=eval_results_list
        )
    else:
        print("retraining mode invalid")
        sys.exit(1)

    process_time = time.time() - process_start_date

    # --- Save settings_log
    settings["Log"]["START_DAYTIME"] = start_date.isoformat()
    settings["Log"]["END_DAYTIME"] = end_date.isoformat()
    settings["Log"]["Processing_TIME"] = process_time
    with open(f"{o_dir_path}/settings_log_edge.json", "w") as f:
        json.dump(settings, f, indent=1)  # type:

    # --- Results processing
    add_results_col = ["nmr_fn_rate", "nmr_benign_rate"]
    add_results_list = []

    sum_fn = np.sum(eval_results_list[:, 5])

    # nmr_flow_num_ratio
    min_max_scaler = MinMaxScaler()
    fn_rate = eval_results_list[:, 5] / sum_fn
    reshaped_fn_rate = fn_rate.reshape(-1, 1)
    scaled_fn_rate = min_max_scaler.fit_transform(reshaped_fn_rate)
    add_results_list.append(scaled_fn_rate.flatten())

    # nmr_benign_ratio
    reshaped_ben_ratio = eval_results_list[:, 14].reshape(-1, 1)
    scaled_ben_ratio = min_max_scaler.fit_transform(reshaped_ben_ratio)
    add_results_list.append(scaled_ben_ratio.flatten())

    # Convert additional results to a 2D array
    add_results_list = np.array(add_results_list).T  # 転置して列形式に変換

    # training results
    tr_results = pd.DataFrame(tr_results_list, columns=t_results_col)
    tr_results.to_csv(os.path.join(o_dir_path, "results_training.csv"), index=False)

    # evaluate results
    eval_results = pd.DataFrame(eval_results_list, columns=eval_results_col)
    add_results = pd.DataFrame(add_results_list, columns=add_results_col)

    # Combine evaluate_results with additional_results
    eval_results = pd.concat([eval_results, add_results], axis=1)
    eval_results.to_csv(os.path.join(o_dir_path, "results_evaluate.csv"), index=False)

    print("\n")
    model.summary()


if __name__ == "__main__":
    main()