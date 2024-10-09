import pickle

# 例としてscikit-learnのモデルを保存する場合
# modelは事前に学習済みのモデル

import paramiko
from scp import SCPClient

# SSH接続のための関数
def create_ssh_client(hostname, port, username, password):
    ssh = paramiko.SSHClient()
    ssh.load_system_host_keys()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname, port=port, username=username, password=password)
    return ssh

# ファイル転送のための関数
def transfer_model(local_file, remote_path, hostname, port, username, password):
    # SSH接続を作成
    ssh = create_ssh_client(hostname, port, username, password)

    # SCPクライアントを使ってファイルを送信
    with SCPClient(ssh.get_transport()) as scp:
        scp.put(local_file, remote_path)

    # SSH接続を閉じる
    ssh.close()

# 使用例
local_file = 'model.pkl'  # pickleで保存したモデルファイル
remote_path = '/remote/path/model.pkl'  # リモート側での保存パス
hostname = '192.168.1.100'  # リモートデバイスのIPアドレス
port = 22  # SSHポート（通常は22）
username = 'your_username'  # SSHユーザー名
password = 'your_password'  # SSHパスワード

# モデルをリモートデバイスに送信
transfer_model(local_file, remote_path, hostname, port, username, password)

