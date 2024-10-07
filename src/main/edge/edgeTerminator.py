import json
import os
import sys
from datetime import datetime, timedelta

import pandas as pd
import pytz
from sklearn.preprocessing import StandardScaler

import modelCreator
import modelSender
import modelTrainer
import modelEvaluator


def set_results_frame(column):
    results_frame = pd.DataFrame(columns=column)
    return results_frame


def is_pass_exist(path):
    if not os.path.exists(path):  # error >> argument 1 name
        print(f"= > cannot find training dataset: {os.path.basename(path)} \n>")
        sys.exit(1)


def output_mkdir(dir_name, settings_file):
    os.makedirs(dir_name)
    os.makedirs(f"{dir_name}/model_weights")
    with open(f'{dir_name}/settings_log.json','w') as f:
        json.dump(settings_file,f,indent=1)


def save_settings_log(settings):
    settings['Log'] = {}
    settings['Log']['INIT_TIME'] = init_time
    settings['Log']['BEGINNING_DAYTIME'] = beginning_daytime.isoformat()
    settings['Log']['END_DAYTIME'] = end_daytime.isoformat()
    with open(f"src/main/edge/outputs/{init_time}_executed/settings_log.json", "w") as f:
        json.dump(settings, f, indent=1)


def save_results(results_list, init_time, results):
    dir_name = f"outputs/{init_time}_executed"
    results = pd.concat([results, pd.DataFrame(results_list, columns=results.columns)])
    results.to_csv(f"{dir_name}/results.csv",index=False)


# ----- Main

if __name__ == "__main__":

    # --- get current time in JST
    jst = pytz.timezone('Asia/Tokyo')
    init_time = datetime.now(jst).strftime("%Y%m%d%H%M%S")

    # --- set results DataFrame
    training_results = set_results_frame(['training_count','training_time','benign_count','malicious_count'])
    training_results_list = training_results.values
    evaluate_results = set_results_frame(['f1_score','precision','recall'])
    evaluate_results_list = evaluate_results.values

    # --- load settings
    settings = json.load(open('src/main/edge/settings.json', 'r'))

    # --- Setting
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = settings['OS']['TF_CPP_MIN_LOG_LEVEL']  # log amount
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = settings['OS']['TF_FORCE_GPU_ALLOW_GROWTH']  # gpu mem limit
    os.environ["CUDA_VISIBLE_DEVICES"] = settings['OS']['CUDA_VISIBLE_DEVICES']  # cpu : -1
    is_pass_exist(settings['Datasets']['Dir_Path']['TRAINING'])
    is_pass_exist(settings['Datasets']['Dir_Path']['TEST'])

    # --- Field
    training_datasets_folder_path = settings['Datasets']['Dir_Path']['TRAINING']
    test_datasets_folder_path = settings['Datasets']['Dir_Path']['TEST']
    beginning_daytime = datetime.strptime(settings['Datasets']['BEGINNING_DAYTIME'],"%Y%m%d%H%M%S")
    days = settings['Datasets']['LearningRange']['DAYS']
    hours = settings['Datasets']['LearningRange']['HOURS']
    minutes = settings['Datasets']['LearningRange']['MINUTES']
    seconds = settings['Datasets']['LearningRange']['SECONDS']
    end_daytime = beginning_daytime + timedelta(days=days,hours=hours,minutes=minutes,seconds=seconds)
    epochs = settings['LearningDefine']['EPOCHS']
    batch_size = settings['LearningDefine']['BATCH_SIZE']
    repeat_count = settings['LearningDefine']['REPEAT_COUNT']
    scaler = StandardScaler()

    # --- Create output directory
    output_dir_name = f"src/main/edge/outputs/{init_time}_executed"
    output_mkdir(output_dir_name, settings)

    # --- Create foundation model
    foundation_model = modelCreator.main()

    # --- Send foundation model to Gateway
    modelSender.main(foundation_model)

    # --- Training model
    model,training_results_list = modelTrainer.main(
        model=foundation_model,
        init_time=init_time,
        training_datasets_folder_path=training_datasets_folder_path,
        scalar=scaler,
        beginning_daytime=beginning_daytime,
        end_daytime=end_daytime,
        repeat_count=repeat_count,
        epochs=epochs,
        batch_size=batch_size,
        results_list=training_results_list
    )

    # --- Evaluate model
    evaluate_results_list = modelEvaluator.main(
        model=model,
        init_time=init_time,
        test_datasets_folder_path=test_datasets_folder_path,
        scalar=scaler,
        beginning_daytime=beginning_daytime,
        end_daytime=end_daytime,
        results_list=evaluate_results_list
    )

    # --- Save settings_log and results
    save_settings_log(settings)
    save_results(training_results_list, init_time, training_results)
    save_results(evaluate_results_list, init_time, evaluate_results)
