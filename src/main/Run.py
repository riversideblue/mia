import time


from ModelFactory import *
from SettingsLoader import *
from SessionController import *


def resolve_input_dim(settings):
    schema = settings.get("FeatureSchema", {})
    mode = schema.get("MODE", "legacy")
    if mode == "split":
        features = schema.get("VECTOR_FEATURES", [])
    elif mode == "legacy":
        features = schema.get("LEGACY_FEATURES", [])
    else:
        raise ValueError(f"Invalid FeatureSchema MODE: {mode}")
    if not features:
        raise ValueError("FeatureSchema must define non-empty feature list.")
    return len(features)

def main():

    # 設定読み込み
    loader = SettingsLoader()

    # セッションの初期化
    session = SessionController(loader)

    input_dim = resolve_input_dim(loader.settings)

    # モデルの初期化
    model_factory = ModelFactory(
        model_code=loader.get('MODEL_CODE'),
        user_dir_path=f"{loader.get('USER_DIR')}",
        foundation_model_path=f"{loader.get('FOUNDATION_MODEL_PATH')}",
        input_dim=input_dim
    )

    # セッションの開始
    session.run(model_factory)

if __name__ == "__main__":
    main()
