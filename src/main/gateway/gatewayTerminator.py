import pickle


# ----- Main

if __name__ == "__main__":

    # リモートデバイスでモデルを読み込む
    with open('/remote/path/model.pkl', 'rb') as f:
        model = pickle.load(f)
