import time


from ModelFactory import *
from SettingsLoader import *
from SessionController import *

def main():

    # 開始時間の記録
    init_time = time.time()

    # 設定読み込み
    loader = SettingsLoader()

    # セッションの初期化
    session = SessionController(loader, init_time)

    # モデルの初期化
    model_factory = ModelFactory(
        model_code=loader.get('MODEL_CODE'),
        foundation_path=f"{loader.get('USER_DIR')}/{loader.get('FOUNDATION_MODEL_PATH')}"
    )

    # セッションの開始
    session.run(model_factory.foundation_model)

if __name__ == "__main__":
    main()