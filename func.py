import streamlit as st
import numpy as np

# データ整理用の設定値
bitsize_arr = [1, 1, 1, 1, 1, 1, 10, 1, 1, 4, 10, 32]  # ログ1の形式 64bit
data_size = 8   # 単位Byte
batch_size = 8   # 単位Byte
chunk_size = 256
offset = 2256

# 関数準備
# バイナリデータを任意のビット形式に変換し、整数値リストとして出力する関数
def reshape_data(data_size, bitsize_arr, data):
    total_bits = data_size * 8
    if sum(bitsize_arr) > total_bits:
        raise ValueError("ビットサイズの合計がデータサイズを超えています。")
    result = []
    current_bit = 0
    for bitsize in bitsize_arr:
        if current_bit + bitsize > total_bits:
            raise ValueError("ビットサイズの合計がデータサイズを超えています。")
        byte_offset = current_bit // 8
        bit_offset = current_bit % 8
        value = 0
        remaining_bits = bitsize
        while remaining_bits > 0:
            bits_to_read = min(8 - bit_offset, remaining_bits)                      # 読み取るビット数を計算
            mask = (1 << bits_to_read) - 1                                          # マスクを作成して特定のビット数を抽出
            value <<= bits_to_read                                                  # 一時的な整数値を左シフトして空間を確保
            value |= (data[byte_offset] >> (8 - bit_offset - bits_to_read)) & mask  # データを復元して value に追加
            remaining_bits -= bits_to_read                                          # 読み取ったビット数を残りから減算
            byte_offset += 1
            bit_offset = 0
        result.append(value)
        current_bit += bitsize
    return result

# 指定のデータ数で1つのまとまりとし、配列の次元を増やす関数(3次元リスト配列)
def split_list_into_arrays(input_list, chunk_size):
    array_list = []
    for i in range(0, len(input_list), chunk_size):
        array_list.append(input_list[i:i+chunk_size])
    return array_list

# 必要データを抽出し N針 x 256 x 12 （1ワーク分）の3次元配列を生成する関数
@st.cache_data
def make_work(uploaded_file, stand):
    uploaded_file.seek(offset)      # ヘッダー分の*バイトをスキップ
    bin_data = uploaded_file.read() # スキップしたデータを読込
    total_data_size = len(bin_data)
    result_batches = []
    
    # bin_dataをbatch_sizeごとにバッチ処理する(2次元リスト配列)
    for i in range(0, total_data_size, batch_size):
        batch_data = bin_data[i:i+batch_size]
        result = reshape_data(data_size, bitsize_arr, batch_data)
        result_batches.append(result)
    
    # 1針目の0度からに設定
    count = 0
    for i in range(512):
        count += 1
        if result_batches[i][1] == 1:   # スタートフラグの立ち上がりを監視
            break
    dim = count - stand                    # 針上は60°なので、43カウント前が約0°
    del result_batches[:dim]
    
    # 256個ずつの配列とする(3次元リスト配列)  [[[],[],[],...,[]],[[],[],[],...,[]],...,[[],[],[],...,[]]]
    result_arrays = split_list_into_arrays(result_batches, chunk_size)
    
    # numpy配列へ変換し、最終針データを削除
    n_array = np.array(result_arrays[:len(result_arrays)-1])
    
    return n_array

# 行列の列で標準偏差を計算する関数
def calculate_std_deviation(matrix):
    std_deviation = np.std(matrix, axis=0)
    return std_deviation

# 閾値に対する大小で色を変える関数
def color_by_threshold(val, upper_threshold, lower_threshold):
    if val < lower_threshold:
        color = 'blue'
    elif val > upper_threshold:
        color = 'red'
    else:
        color = 'gray'
    return f'color: {color}'