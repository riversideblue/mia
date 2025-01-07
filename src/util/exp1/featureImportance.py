import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ディレクトリのパス
all_dir_path = "/mnt/nas0/g005/murasemaru/exp/1_DataAnalytics/drift/unproc"

# 相関係数を保存する辞書
corrs = {}
for dir_name in os.listdir(all_dir_path):
    if not dir_name == 'res_img':
        dir_path = os.path.join(all_dir_path, dir_name)
        if dir_name not in corrs:
            corrs[dir_name] = {}
        if os.path.isdir(dir_path):  # ディレクトリであることを確認
            for file in os.listdir(dir_path):
                if file.endswith('.csv'):
                    file_path = os.path.join(dir_path, file)
                    df = pd.read_csv(file_path)
                    if file not in corrs:
                        corrs[dir_name][file] = {}
                    # 各列の相関を計算
                    for column in df.columns:
                        if (
                            column not in ["mean_dis", "w_mean_dis", "ks_mean_dis","w_label","ks_label"]
                            and pd.api.types.is_numeric_dtype(df[column])
                        ):
                            
                            corrs[dir_name][file][column] = df[column].corr(df["mean_dis"])
                    
                    print(f"Processing: {file_path}")

x_keys = list(corrs.keys())
y_keys = list(corrs[x_keys[0]].keys())
z_keys = list(corrs[x_keys[0]][y_keys[0]].keys())

print(x_keys)
print(y_keys)
print(z_keys)

# 辞書を三次元配列に変換
three_d_array = np.array([
    [
        [corrs[x][y][z] for z in z_keys]
        for y in y_keys
    ]
    for x in x_keys
])

print(f"axis0(region): {np.shape(three_d_array)[0]}")
print(f"axis1(window_size): {np.shape(three_d_array)[1]}")
print(f"axis2(features): {np.shape(three_d_array)[2]}")

# pop次元を圧縮した reg - feature の mean_dis相関行列
reg_corr = np.mean(three_d_array, axis=0)
# reg次元を圧縮した pop - feature の mean_dis相関行列
pop_corr = np.mean(three_d_array,axis=1)
# pop次元，reg次元を圧縮した pop - feature の mean_dis相関行列
res_corr = np.mean(three_d_array,axis=0,1)

print(reg_corr)
print(pop_corr)

mean_reg_corr = []
mean_pop_corr = []
mean_res_corr = []

for z_key in z_keys:
    p = z_key.split('_',1)
    if p[0] != 'w' or p != 'ks':
        mean_reg_corr.append(reg_corr[:,0])
        mean_pop_corr.append(pop_corr[:,0])
        mean_res_corr.append(res_corr[0])
    elif p[0] == 'w':
        target_key = f"ks_{p[1]}"
        target_index = z_keys.index(target_key)
        ean_reg_corr.append(reg_corr[:,target_index])
        mean_pop_corr.append(pop_corr[:,target_index])
        mean_res_corr.append(res_corr[target_index])

# ヒートマップの作成
sns.heatmap(mean_reg_corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("mean reg corr")
plt.show()

# ヒートマップの作成
sns.heatmap(mean_pop_corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("mean pop corr")
plt.show()

plt.hist(mean_res_corr, bins=5, edgecolor='black')
plt.title("mean res corr")
plt.xlabel("Feature")
plt.ylabel("Frequency")
