import time


from ModelFactory import *
from SettingsLoader import *
from SessionController import *

def main():

    # 設定読み込み
    loader = SettingsLoader()

    # セッションの初期化
    session = SessionController(loader)

    # モデルの初期化
    model_factory = ModelFactory(
        model_code=loader.get('MODEL_CODE'),
        user_dir_path=f"{loader.get('USER_DIR')}",
        foundation_model_path=f"{loader.get('FOUNDATION_MODEL_PATH')}"
    )

    # セッションの開始
    session.run(model_factory.foundation_model)

if __name__ == "__main__":
    main()