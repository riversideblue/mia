import json
import os
from datetime import datetime

import pytz

import modelCreator
import modelSender
import modelTrainer
import modelEvaluator

def output_mkdir(dir_name, settings_file):

    os.makedirs(dir_name)
    os.makedirs(f"{dir_name}/model_weights")
    with open(f'{dir_name}/settings_log.json','w') as f:
        json.dump(settings_file,f,indent=1)

# ----- Main

if __name__ == "__main__":

    # --- get current time in JST
    jst = pytz.timezone('Asia/Tokyo')
    init_time = datetime.now(jst).strftime("%Y%m%d%H%M%S")

    # --- prepare settings and results
    settings = json.load(open("settings.json", "r"))
    output_dir_name = f"outputs/{init_time}_executed"
    output_mkdir(output_dir_name, settings)

    # --- Create foundation model
    foundation_model = modelCreator.main()

    # --- Send foundation model to Gateway
    modelSender.main(foundation_model)

    # --- Training model
    model = modelTrainer.main(foundation_model,init_time,settings)

    # --- Evaluate model
    modelEvaluator.main(model,init_time,settings)