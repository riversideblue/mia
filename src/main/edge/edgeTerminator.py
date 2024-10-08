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


def output_mkdir(dir_path, settings_file):
    os.makedirs(dir_path)
    os.makedirs(f"{dir_path}/model_weights")
    with open(f'{dir_path}/settings_log.json','w') as f:
        json.dump(settings_file,f,indent=1) # type: ignore


def save_settings_log(settings_log, output_dir):

    with open(f"{output_dir}/settings_log.json", "w") as f:
        json.dump(settings_log, f, indent=1) # type: ignore


def save_results(training_list, training, evaluate_list, evaluate, output_dir):
    training = pd.concat([training, pd.DataFrame(training_list, columns=training.columns)])
    training.to_csv(f"{output_dir}/results_training.csv",index=False)
    evaluate = pd.concat([evaluate, pd.DataFrame(evaluate_list, columns=evaluate.columns)])
    evaluate.to_csv(f"{output_dir}/results_evaluate.csv", index=False)


# ----- Main

if __name__ == "__main__":

    # --- load settings
    settings = json.load(open('src/main/edge/settings.json', 'r'))
    settings['Log'] = {}

    # --- get current time in JST
    jst = pytz.timezone('Asia/Tokyo')
    init_time = datetime.now(jst).strftime("%Y%m%d%H%M%S")
    settings['Log']['INIT_TIME'] = init_time

    # --- set results DataFrame
    training_results = set_results_frame(['training_count','training_time','benign_count','malicious_count'])
    training_results_list = training_results.values
    evaluate_results = set_results_frame(['f1_score','precision','recall'])
    evaluate_results_list = evaluate_results.values

    # --- Setting
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = settings['OS']['TF_CPP_MIN_LOG_LEVEL']  # log amount
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = settings['OS']['TF_FORCE_GPU_ALLOW_GROWTH']  # gpu mem limit
    os.environ["CUDA_VISIBLE_DEVICES"] = settings['OS']['CUDA_VISIBLE_DEVICES']  # cpu : -1

    # --- Field
    settings['Log']['Training'] = {}
    training_datasets_folder_path = settings['Training']['DATASETS_FOLDER_PATH']
    training_beginning_daytime = datetime.strptime(settings['Training']['BEGINNING_DAYTIME'],"%Y%m%d%H%M%S")
    settings['Log']['Training']['BEGINNING_DAYTIME'] = training_beginning_daytime.isoformat()
    training_days = settings['Training']['TargetRange']['DAYS']
    training_hours = settings['Training']['TargetRange']['HOURS']
    training_minutes = settings['Training']['TargetRange']['MINUTES']
    training_seconds = settings['Training']['TargetRange']['SECONDS']
    training_end_daytime = training_beginning_daytime + timedelta(
        days=training_days,
        hours=training_hours,
        minutes=training_minutes,
        seconds=training_seconds
    )
    settings['Log']['Training']['END_DAYTIME'] = training_end_daytime.isoformat()
    epochs = settings['Training']['LearningDefine']['EPOCHS']
    batch_size = settings['Training']['LearningDefine']['BATCH_SIZE']
    repeat_count = settings['Training']['LearningDefine']['REPEAT_COUNT']

    settings['Log']['Evaluate'] = {}
    evaluate_datasets_folder_path = settings['Evaluate']['DATASETS_FOLDER_PATH']
    evaluate_beginning_daytime = datetime.strptime(settings['Evaluate']['BEGINNING_DAYTIME'],"%Y%m%d%H%M%S")
    settings['Log']['Evaluate']['BEGINNING_DAYTIME'] = evaluate_beginning_daytime.isoformat()
    evaluate_days = settings['Evaluate']['TargetRange']['DAYS']
    evaluate_hours = settings['Evaluate']['TargetRange']['HOURS']
    evaluate_minutes = settings['Evaluate']['TargetRange']['MINUTES']
    evaluate_seconds = settings['Evaluate']['TargetRange']['SECONDS']
    evaluate_end_daytime = evaluate_beginning_daytime + timedelta(
        days=evaluate_days,
        hours=evaluate_hours,
        minutes=evaluate_minutes,
        seconds=evaluate_seconds
    )
    settings['Log']['Evaluate']['END_DAYTIME'] = evaluate_end_daytime.isoformat()
    scaler = StandardScaler()

    # --- Pass check
    is_pass_exist(training_datasets_folder_path)
    is_pass_exist(evaluate_datasets_folder_path)

    # --- Create output directory
    output_dir_path:str = f"src/main/edge/outputs/{init_time}_executed"
    output_mkdir(output_dir_path, settings)

    # --- Create foundation model
    foundation_model = modelCreator.main()

    # --- Send foundation model to Gateway
    modelSender.main(foundation_model)

    # --- Training model
    model,training_results_list = modelTrainer.main(
        model=foundation_model,
        init_time=init_time,
        datasets_folder_path=training_datasets_folder_path,
        scalar=scaler,
        beginning_daytime=training_beginning_daytime,
        end_daytime=training_end_daytime,
        repeat_count=repeat_count,
        epochs=epochs,
        batch_size=batch_size,
        results_list=training_results_list
    )

    print(model)

    # --- Evaluate model
    evaluate_results_list = modelEvaluator.main(
        model=model,
        init_time=init_time,
        datasets_folder_path=evaluate_datasets_folder_path,
        scalar=scaler,
        beginning_daytime=evaluate_beginning_daytime,
        end_daytime=evaluate_end_daytime,
        results_list=evaluate_results_list
    )

    # --- Save settings_log and results
    save_settings_log(settings,output_dir_path,)
    save_results(
        training_list=training_results_list,
        training=training_results,
        evaluate_list=evaluate_results_list,
        evaluate=evaluate_results,
        output_dir=output_dir_path
    )