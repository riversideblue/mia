import asyncio
import json
import os
import sys
from datetime import datetime, timedelta

import pandas as pd
import pytz
from sklearn.preprocessing import StandardScaler

from main.other import modelCreator
from main.edge import driftDetector, modelSender, modelTrainer, trafficServer, featuresExtractor


def is_pass_exist(path):
    if not os.path.exists(path):  # error >> argument 1 name
        print(f"Cannot find training dataset: {os.path.basename(path)}")
        sys.exit(1)


async def training_activate(dynamic_mode, static_interval, model, output_dir_path, datasets_folder_path, scaler, beginning_daytime, end_daytime, repeat_count, epochs, batch_size, results, results_list):
    if not dynamic_mode:
        while True:
            modelTrainer.main(
                model=model,
                output_dir_path=output_dir_path,
                datasets_folder_path=datasets_folder_path,
                scalar=scaler,
                beginning_daytime=beginning_daytime,
                end_daytime=end_daytime,
                repeat_count=repeat_count,
                epochs=epochs,
                batch_size=batch_size,
                results=results,
                results_list=results_list
            )
            await asyncio.sleep(static_interval)
    else:
        print("dynamic_mode on")
        driftDetector.main(model)


async def traffic_server_activate():
    trafficServer.main()


# ----- Main

async def main():
    # --- Load settings
    settings = json.load(open("src/main/edge/settings.json", "r"))
    settings["Log"] = {}

    # --- Get current time in JST
    jst = pytz.timezone("Asia/Tokyo")
    init_time: str = datetime.now(jst).strftime("%Y%m%d%H%M%S")
    settings["Log"]["INIT_TIME"] = init_time

    # --- Set results DataFrame
    results = pd.DataFrame(columns=["training_count", "training_time", "benign_count", "malicious_count"])
    results_list = results.values

    # --- Setting
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = settings["OS"]["TF_CPP_MIN_LOG_LEVEL"]  # log amount
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = settings["OS"]["TF_FORCE_GPU_ALLOW_GROWTH"]  # gpu mem limit
    os.environ["CUDA_VISIBLE_DEVICES"] = settings["OS"]["CUDA_VISIBLE_DEVICES"]  # cpu : -1

    # --- Field
    settings["Log"]["Training"] = {}
    datasets_folder_path: str = settings["Training"]["DATASETS_FOLDER_PATH"]
    beginning_daytime = datetime.strptime(settings["Training"]["BEGINNING_DAYTIME"], "%Y-%m-%d %H:%M:%S")
    settings["Log"]["Training"]["BEGINNING_DAYTIME"] = beginning_daytime.isoformat()
    days: int = settings["Training"]["TargetRange"]["DAYS"]
    hours: int = settings["Training"]["TargetRange"]["HOURS"]
    minutes: int = settings["Training"]["TargetRange"]["MINUTES"]
    seconds: int = settings["Training"]["TargetRange"]["SECONDS"]
    end_daytime: datetime = beginning_daytime + timedelta(
        days=days,
        hours=hours,
        minutes=minutes,
        seconds=seconds
    )
    settings["Log"]["Training"]["END_DAYTIME"] = end_daytime.isoformat()
    epochs: int = settings["Training"]["LearningDefine"]["EPOCHS"]
    batch_size: int = settings["Training"]["LearningDefine"]["BATCH_SIZE"]
    repeat_count: int = settings["Training"]["LearningDefine"]["REPEAT_COUNT"]

    online_mode: bool = settings["Training"]["RetrainingCycle"]["ONLINE_MODE"]
    dynamic_mode: bool = settings["Training"]["RetrainingCycle"]["DYNAMIC_MODE"]
    static_interval: int = settings["Training"]["RetrainingCycle"]["STATIC_INTERVAL"]

    scaler = StandardScaler()
    is_pass_exist(datasets_folder_path)

    # --- Create output directory
    output_dir_path: str = f"src/main/edge/outputs/{init_time}_executed"
    os.makedirs(output_dir_path)
    os.makedirs(f"{output_dir_path}/model_weights")

    # --- Create foundation model
    model = modelCreator.main()

    # --- Online mode or not

    if not online_mode:
        print("\n- offline mode activated")
        modelTrainer.main(
            model=model,
            output_dir_path=output_dir_path,
            datasets_folder_path=datasets_folder_path,
            scalar=scaler,
            beginning_daytime=beginning_daytime,
            end_daytime=end_daytime,
            repeat_count=repeat_count,
            epochs=epochs,
            batch_size=batch_size,
            results=results,
            results_list=results_list
        )

    else:
        print("\n- online mode activated")
        # --- Send foundation model to Gateway
        modelSender.main(model)

        # --- Wait reserve traffic data

        # --- Convert pcap to csv
        featuresExtractor.online()
        # --- Retraining model
        await asyncio.gather(
            traffic_server_activate(),
            training_activate(
                dynamic_mode=dynamic_mode,
                static_interval=static_interval,
                model=model,
                output_dir_path=output_dir_path,
                datasets_folder_path=datasets_folder_path,
                scaler=scaler,
                beginning_daytime=beginning_daytime,
                end_daytime=end_daytime,
                repeat_count=repeat_count,
                epochs=epochs,
                batch_size=batch_size,
                results=results,
                results_list=results_list  # この後ゲートウェイへのモデル送信を行う
            )
        )

    # --- Save settings_log and results
    with open(f"{output_dir_path}/settings_log_edge.json", "w") as f:
        json.dump(settings, f, indent=1)  # type: ignore

if __name__ == "__main__":
    asyncio.run(main())
