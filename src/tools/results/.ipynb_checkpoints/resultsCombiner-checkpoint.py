import pandas as pd

# ファイルの読み込み
csv1 = "/home/murasemaru/nids-cdd/experiment/ex2-fmda-nt/1-wt2022-dy/results_evaluate.csv"  # CSVファイル1のパス
csv2 = "/home/murasemaru/nids-cdd/experiment/ex2-fmda-nt/1-wt2022-st/20241210043940/results_evaluate.csv"  # CSVファイル2のパス

df1 = pd.read_csv(csv1)  # ファイル1をデータフレームに読み込み
df2 = pd.read_csv(csv2)  # ファイル2をデータフレームに読み込み

# 特定の行を指定
row_index_t = 0
row_index_1 = 10  # CSV1から結合したい行番号（0始まり）
row_index_2 = 10  # CSV2から結合したい行番号（0始まり）
row_index_3 = 16

# 行を取得
timestamp = df1.iloc[:,row_index_t]
row1 = df1.iloc[:,row_index_1]  # CSV1の指定行
row2 = df2.iloc[:,row_index_2]  # CSV2の指定行
row3 = df2.iloc[:,row_index_3]  # CSV2の指定行

# 水平方向に結合（DataFrameとして保存）
combined_row = pd.concat([timestamp,row1, row2, row3], axis=1)

# 結果を確認
print(combined_row)

# 結果をCSVとして保存する場合
combined_row.to_csv("combined_row.csv", index=False, header=["daytime","dynamic_accuracy","static_accuracy","nmr_benign_rate"])
