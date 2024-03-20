#v029 ワーク数指定の箇所の上限が針数になっていたので修正
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from scipy.stats import skew, kurtosis
import base64

# streamlitの表示用リストとタブ
options = ['糸切れ','目飛び','締り1','締り2','締り3']
kishu = ['小型BAS','大型BAS','全回転BAS','BUF','DA',7300]
jikan = ['1us', '1ms', '333ns']
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["データアップ", "総和値算出", "不良フラグ表示", "速度フラグ表示", "重ねて表示", "外れ値解析", "σ確認", "3Dグラフ"])

# サイドバー表示 (機種設定：速度計算は未対応）
type = st.sidebar.selectbox('機種設定', kishu)
count_unit = st.sidebar.selectbox('1カウントの時間', jikan)
if count_unit == '1us':
    clk = 1000
elif count_unit == '1ms':
    clk = 1000000
else:
    clk = 333
st.sidebar.write("3G : 1us")
st.sidebar.write("2G : 333ns")

# データ整理用の設定値
bitsize_arr = [1, 1, 1, 1, 1, 1, 10, 1, 1, 4, 10, 32]  # ログ1の形式 64bit
data_size = 8   # 単位Byte
batch_size = 8   # 単位Byte
chunk_size = 256
offset = 2256

if type == 'BUF':
    stand = 35
    ten_st = 29
    ten_ed = 57
    km_st = 220
    km_ed = 248
    d_st = 137
    d_ed = 151
    dd = 2
elif type == 'DA':
    stand = 253
    ten_st = 29
    ten_ed = 57
    km_st = 220
    km_ed = 248
    d_st = 131
    d_ed = 145
    dd = 2
elif type == '小型BAS':
    stand = 42
    ten_st = 32
    ten_ed = 60
    km_st = 220
    km_ed = 248
    d_st = 128
    d_ed = 156
    dd = 1
elif type == '大型BAS':
    stand = 42
    ten_st = 32
    ten_ed = 60
    km_st = 220
    km_ed = 248
    d_st = 128
    d_ed = 156
    dd = 1
elif type == '全回転BAS':
    stand = 42
    ten_st = 22
    ten_ed = 50
    km_st = 220
    km_ed = 248
    d_st = 128
    d_ed = 156
    dd = 1
else:
    stand = 17
    ten_st = 36
    ten_ed = 64
    km_st = 220
    km_ed = 248
    d_st = 128
    d_ed = 156
    dd = 1


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

# データフレームをCSVファイルに保存する関数
def save_to_csv(df, file_path):
    df.to_csv(file_path, index=False)
    st.success(f"CSVファイルが {file_path} に保存されました。")

# csv出力する関数
def csvout(data, name, filename):
    # CSV出力のためのボタン
    if st.button(f"{name}", help="グラフのデータをCSVファイルに出力", type='primary'):
        # CSV出力のためのデータフレームを作成
        df_export = pd.DataFrame({
            'DATA': data,
        })
        csv_file = df_export.to_csv(index=False)
        b64 = base64.b64encode(csv_file.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">{name}</a>'
        st.markdown(href, unsafe_allow_html=True)

# データフレームをCSVファイルに保存する関数
def save_to_csv(df, file_path):
    df.to_csv(file_path, index=False)
    st.success(f"CSVファイルが {file_path} に保存されました。")

# csv出力する関数2
def csvouts(data_list, name, filename):
    # CSV出力のためのボタン
    if st.button(f"{name}", help="グラフのデータをCSVファイルに出力", type='primary'):
        # 3次元アレイを2次元に変換
        flattened_data = data_list.reshape(data_list.shape[0], -1)

        # CSV出力のためのデータフレームを作成
        df_export = pd.DataFrame(flattened_data.T, columns=[f"work{i+1}" for i in range(flattened_data.shape[0])])

        # CSVファイルに保存
        save_to_csv(df_export, f"{filename}.csv")

        # ダウンロードリンクの生成
        csv_file = df_export.to_csv(index=False)
        b64 = base64.b64encode(csv_file.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">{name}</a>'
        st.markdown(href, unsafe_allow_html=True)


# 行列の列で標準偏差を計算する関数
def calculate_std_deviation(matrix):
    std_deviation = np.std(matrix, axis=0)
    return std_deviation

# メインプログラム
if __name__ == "__main__":
    
    with tab1:
        st.header("SDD LOG解析", divider='red')
        st.write("### ▼データアップロード")
        uploaded_files = st.file_uploader("BINデータをアップロードしてください", type=['bin', 'sddb'], accept_multiple_files=True)
        
        if uploaded_files:
            n_arrays = []   # 1ワーク分の3次元配列
            for uploaded_file in uploaded_files:
                n_array = make_work(uploaded_file, stand)
                n_arrays.append(n_array)
            w_array = np.stack(n_arrays, axis=0)    # 複数ワーク分の4次元配列（同じ針数のワークしか受け付けない）
            file_no = st.number_input('何番目のファイルを分析しますか？', min_value=1, step=1)
            
            # 針数毎のグラフを重ねて表示
            angle = np.arange(0, 360, 1.40625)  # 角度に変換
            
            # データの作成
            data1 = []
            for i in range(len(n_array)):
                random_color = tuple(np.random.choice(range(256), size=3)) # RGBの範囲内でランダムな色を生成
                trace = go.Scatter(
                    x = angle,
                    y = w_array[file_no-1,i,:,6],
                    mode = 'lines',
                    name = f"{i+1}針",
                    line = dict(color = 'rgb'+str(random_color), width = 2) # 'rgb()'形式の文字列に変換
                )
                data1.append(trace)
            # レイアウトの作成
            layout1 = go.Layout(
                title = 'Tension per stitch',
                xaxis = dict(title = 'Angle'),
                yaxis = dict(title = 'Tension'),
                showlegend = True
            )
            # 全張力横並び表示
            xa = np.array(range(len(n_array)*256))
            flat = w_array[file_no-1,:,:,6].flatten()
            # データの作成
            data2 = go.Scatter(x=xa, y=flat, line=dict(color="mediumblue"), name='Array #{}'.format(i))
            # グラフのタイトルと軸ラベルの設定
            layout2 = go.Layout(
                title='Tension per Work',
                xaxis=dict(
                    title='Count',
                    tickvals=list(range(0, len(flat), 256)),  # 横軸の目盛りの位置
                    ticktext=[i+1 for i in range(0,len(flat) // 256)]  # 横軸の目盛りのラベル
                ),
                yaxis=dict(title='Tension'),
            )
            
            if st.button("全体波形表示", help="時系列表示と重ねた波形を表示", type='primary'):    # on_clickで関数を指定できる
                # フィギュアの作成
                fig1 = go.Figure(data=data1, layout=layout1)
                # フィギュアの表示
                st.plotly_chart(fig1, use_container_width=True)
                # フィギュアの作成
                fig2 = go.Figure(data=data2, layout=layout2)
                # フィギュアの表示
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.write('アップロードしたら「全体波形表示」をクリック!!')
        else:
            st.write('### データをアップロードしてください')

    with tab2:
        if uploaded_files:
            st.header("SDD LOG解析", divider='red')
            st.write("### ▼総和値算出")
            st.write("解析する張力総和区間を設定してください")
            col1, col2 = st.columns(2)
            with col1:
                stt = st.number_input('開始位相', value=ten_st)
                st.markdown(f"<span style='font-size: 12px;'>換算</span> {stt * 1.40625}°", unsafe_allow_html=True)  #カウント数を角度に換算
            with col2:
                end = st.number_input('終了位相', value=ten_ed)
                st.markdown(f"<span style='font-size: 12px;'>換算</span> {end * 1.40625}°", unsafe_allow_html=True)
            
            st.write("出会い張力総和区間を設定してください")
            col3, col4 = st.columns(2)
            with col3:
                dstt = st.number_input('出会い開始位相', value=d_st)
                st.markdown(f"<span style='font-size: 12px;'>換算</span> {dstt * 1.40625}°", unsafe_allow_html=True)
    
            with col4:
                dend = st.number_input('出会い終了位相', value=d_ed)
                st.markdown(f"<span style='font-size: 12px;'>換算</span> {dend * 1.40625}°", unsafe_allow_html=True)
            
            x = np.array(range(len(n_array)))  # 横軸針数
            sowa = np.arange(len(n_array))  # 1針毎の総和値を入れる配列を作成
            for i in range(len(n_array)):  # 指定区間の総和値をsowa配列に保存する
                sowa[i] = np.sum(w_array[file_no-1,i,stt:end,6]) - np.sum(w_array[file_no-1,i,dstt:dend,6])*dd  # 出会い区間128:156
            #st.write(np.sum(w_array[file_no-1,i,dstt:dend,6])*dd)
            # データの作成
            data3 = go.Scatter(x=x+1, y=sowa, mode='lines+markers', line=dict(color="#757575"),  name='総和値'.format(i))
            # レイアウトの作成
            layout = go.Layout(
                title='Tension per a part of stitch',
                xaxis=dict(title='Nd Count'),
                yaxis=dict(title='Tension'),
                showlegend=True
            )
            if st.button("算出波形出力", help="指定区間総和値を針数毎に表示", type='primary'):
                # フィギュアの作成
                fig3 = go.Figure(data=data3, layout=layout)
                # フィギュアの表示
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.write('データアップロード後に「算出波形出力」をクリック!!')
        else:
            st.write('### データをアップロードしてください')

    with tab3:
        if uploaded_files:
            st.header("SDD LOG解析", divider='red')
            st.write("### ▼不良フラグ表示")
            option = st.selectbox('確認する不良を選択してください', options)
            if option == '糸切れ':
                defect_type = 3
            if option == '目飛び':
                defect_type = 4
            if option == '締り1':
                defect_type = 5
            if option == '締り2':
                defect_type = 7
            if option == '締り3':
                defect_type = 8
            x = np.array(range(len(n_array)))  # 横軸針数
            flag = np.arange(len(n_array))  # 1針毎の総和値を入れる配列を作成
            for i in range(len(n_array)):  # 指定区間の総和値をflag配列に保存する
                flag[i] = np.sum(w_array[file_no-1,i,:,defect_type])
            data4 = go.Scatter(x=x+1, y=flag, mode='lines+markers', line=dict(color="#FF5722"), name='不良フラグ'.format(i), yaxis='y2')
            # レイアウトの作成
            layout = go.Layout(
                title='Defect per stitch',
                xaxis=dict(title='Nd Count'),
                yaxis=dict(title='Flag'),
                showlegend=True
            )
            if st.button("不良個所表示", help="指定不良の針数を表示", type='primary'):
                # フィギュアの作成
                fig4 = go.Figure(data=data4, layout=layout)
                # フィギュアの表示
                st.plotly_chart(fig4, use_container_width=True)
            else:
                st.write('データアップロード後に「不良個所表示」をクリック!!')
        else:
            st.write('### データをアップロードしてください')
        
    with tab4:
        if uploaded_files:
            st.header("SDD LOG解析", divider='red')
            st.write("### ▼速度フラグ表示")
            col1, col2 = st.columns(2)
            with col1:
                flagH = st.number_input('加速判定速度', value=100)
            with col2:
                flagL = st.number_input('減速判定速度', value=100)
            x = np.array(range(len(n_array)))  # 横軸針数
            speed = np.arange(len(n_array))
            speed[0] = 0  # 0rpm
            speed_dim = np.arange(len(n_array))
            speed_dim[0] = 0
            speed_flag = np.arange(len(n_array))
            for i in range(1,len(n_array)):  # 指定区間の総和値をspeed配列に保存する
                speed[i] = 60000000000 // (clk * (w_array[file_no-1,i,127,11] - w_array[file_no-1,i-1,127,11]))  # 10:フリーランカウンタ
                speed_dim[i] = speed[i] - speed[i-1]
                if speed_dim[i] > flagH :
                    speed_flag[i] = 1
                elif speed_dim[i] < 0 and abs(speed_dim[i]) > flagL :
                    speed_flag[i] = -1
                else :
                    speed_flag[i] = 0
            # データの作成
            data5 = go.Scatter(x=x+1, y=speed, mode='lines+markers', line=dict(color="#6495ED"), name='縫製速度'.format(i), yaxis='y1')
            data6 = go.Scatter(x=x+1, y=speed_flag, mode='lines+markers', line=dict(color="#AAA5D1"), name='加減速フラグ'.format(i), yaxis='y2')
            # レイアウトの設定
            layout = go.Layout(
                title='Speed per stitch',
                xaxis=dict(title='Nd Count'),
                yaxis=dict(title='Speed[rpm]'),
                showlegend=True
            )
            if st.button("速度フラグ表示", help="針数毎の速度と速度フラグを表示", type='primary'):
                # フィギュアの作成
                fig5 = go.Figure(data=[data5, data6], layout=layout)
                fig5.update_layout(yaxis1=dict(side='left'), yaxis2=dict(side='right', overlaying = 'y'))
                # フィギュアの表示
                st.plotly_chart(fig5, use_container_width=True)
            else:
                st.write('データアップロード後に「速度フラグ表示」をクリック!!')
        else:
            st.write('### データをアップロードしてください')
        
    with tab5:
        if uploaded_files:
            st.header("SDD LOG解析", divider='red')
            st.write("### ▼波形を重ねて表示")
            Sowa = st.checkbox('総和値', key='-Sowa-')
            Flag = st.checkbox('不良フラグ', key='-Flag-')
            Speed = st.checkbox('速度フラグ', key='-Speed-')
            # レイアウトの設定
            layout = go.Layout(
                title='Analys',
                xaxis=dict(title='Nd Count'),
                yaxis=dict(title='Tens, speed, flags'),
                showlegend=True
            )
            # data7に選択された波形データを格納
            data7 = []
            if Sowa == True:
                data7.extend([data3])
            if Flag == True:
                data7.extend([data4])
            if Speed == True:
                data7.extend([data5])
                data7.extend([data6])
            # データの作成
            data1 = []
            for i in range(len(n_array)):
                random_color = tuple(np.random.choice(range(256), size=3)) # RGBの範囲内でランダムな色を生成
                trace = go.Scatter(
                    x = angle,
                    y = w_array[file_no-1,i,:,6],
                    mode = 'lines',
                    name = f"{i+1}針",
                    line = dict(color = 'rgb'+str(random_color), width = 2) # 'rgb()'形式の文字列に変換
                )
                data1.append(trace)
            # レイアウトの作成
            layout1 = go.Layout(
                title = 'Tension per stitch',
                xaxis = dict(title = 'Angle'),
                yaxis = dict(title = 'Tension'),
                showlegend = True
            )
            # 全張力横並び表示
            xa = np.array(range(len(n_array)*256))
            flat = w_array[file_no-1,:,:,6].flatten()
            # データの作成
            data2 = go.Scatter(x=xa, y=flat, line=dict(color="mediumblue"), name='Array #{}'.format(i))
            # グラフのタイトルと軸ラベルの設定
            layout2 = go.Layout(
                title='Tension per Work',
                xaxis=dict(
                    title='Count',
                    tickvals=list(range(0, len(flat), 256)),  # 横軸の目盛りの位置
                    ticktext=[i+1 for i in range(0,len(flat) // 256)]  # 横軸の目盛りのラベル
                ),
                yaxis=dict(title='Tension'),
            )

            if st.button("重ねて表示", help="選択した波形を重ねて表示", type='primary'):
                # フィギュアの作成
                fig6 = go.Figure(data=data7, layout=layout)
                fig6.update_layout(yaxis1=dict(side='left'), yaxis2=dict(side='right', overlaying = 'y'))
                # フィギュアの表示
                st.plotly_chart(fig6, use_container_width=True)
                #if st.button("全体波形", help="時系列表示と重ねた波形を表示", type='primary'):
                # フィギュアの作成
                fig1 = go.Figure(data=data1, layout=layout1)
                # フィギュアの表示
                st.plotly_chart(fig1, use_container_width=True)
                # フィギュアの作成
                fig2 = go.Figure(data=data2, layout=layout2)
                # フィギュアの表示
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.write('データアップロード後に「重ねて表示」をクリック!!')
        else:
            st.write('### データをアップロードしてください')
    
    with tab6:
        if uploaded_files:
            st.header("SDD LOG解析", divider='red')
            st.write("### ▼外れ値解析")
            st.write("解析する張力総和区間を設定してください")
            col1, col2 = st.columns(2)
            with col1:
                w_stt = st.number_input('開始位相.', value=ten_st)
                st.markdown(f"<span style='font-size: 12px;'>換算</span> {w_stt * 1.40625}°", unsafe_allow_html=True)  #カウント数を角度に換算
            with col2:
                w_end = st.number_input('終了位相.', value=ten_ed)
                st.markdown(f"<span style='font-size: 12px;'>換算</span> {w_end * 1.40625}°", unsafe_allow_html=True)
            
            st.write("出会い張力総和区間を設定してください")
            col3, col4 = st.columns(2)
            with col3:
                w_dstt = st.number_input('出会い開始位相.', value=d_st)
                st.markdown(f"<span style='font-size: 12px;'>換算</span> {w_dstt * 1.40625}°", unsafe_allow_html=True)
    
            with col4:
                w_dend = st.number_input('出会い終了位相.', value=d_ed)
                st.markdown(f"<span style='font-size: 12px;'>換算</span> {w_dend * 1.40625}°", unsafe_allow_html=True)
            
            w_num = st.number_input(f'合算時ワーク数を設定してください。アップロード中の総ワーク数：{len(w_array)}', min_value=1, step=1, value=50)
            
            x = np.array(range(len(w_array)))  # 横軸ワーク数
            nd = st.number_input('何針目を解析しますか？', min_value=1, max_value=len(n_array) , step=1)
            ndsowa = np.arange(len(w_array))  # 各ワークのnd針目の総和値を入れる配列を作成
            for n in range(len(w_array)):  # 指定区間の総和値をsowa配列に保存する
                ndsowa[n] = np.sum(w_array[n,nd,w_stt:w_end,6]) - np.sum(w_array[n,nd,w_dstt:w_dend,6])*dd  # 出会い区間128:156
            # データの作成
            data8 = go.Scatter(x=x+1, y=ndsowa, mode='lines+markers', line=dict(color="#77AF9C"),  name='総和値'.format(i))
            # レイアウトの作成
            layout8 = go.Layout(
                title='総和張力',
                xaxis=dict(title='Work Count'),
                yaxis=dict(title='Section Tension'),
                showlegend=True
            )

            # ヒストグラムを作成
            hist = px.histogram(ndsowa, nbins=15, title=f'{nd}針目のHistogram', labels={'value': 'tension', 'count': 'Frequency'}, color_discrete_sequence=["#77AF9C"])
            # 歪度の計算
            skew = skew(ndsowa)
            # 尖度の計算
            kurt = kurtosis(ndsowa)

            # 標準偏差の移動平均
            std_devs1 = []
            for i in range(len(w_array)-w_num):
                std_dev = np.std(ndsowa[i:i+w_num])  # 標準偏差を計算します
                std_devs1.append(std_dev)  # 結果をリストに追加します
            # データの作成
            data9 = go.Scatter(x=x+1, y=std_devs1, mode='lines+markers', line=dict(color="#77AF9C"),  name='STD1'.format(i))
            # レイアウトの作成
            layout9 = go.Layout(
                title=f'{w_num}ワーク毎の移動STD',
                xaxis=dict(title='work'),
                yaxis=dict(title='std'),
                showlegend=True
            )

            # 10iワーク毎の標準偏差
            std_devs2 = []
            for i in range(int(len(w_array)/10)):
                std_dev2 = np.std(ndsowa[0:(i+1)*10])  # 標準偏差を計算します
                std_devs2.append(std_dev2)  # 結果をリストに追加します
            # データの作成
            data10 = go.Scatter(x=x+1, y=std_devs2, mode='lines+markers', line=dict(color="#77AF9C"),  name='STD2'.format(i))
            # レイアウトの作成
            layout10 = go.Layout(
                title='10ワークずつ増やしたSTD',
                xaxis=dict(title='1/10 work'),
                yaxis=dict(title='std'),
                showlegend=True
            )

            # 標準偏差の合算
            std_devs3 = []
            for i in range(1,len(w_array)):
                if i % w_num == 0:
                    std_dev3 = np.std(ndsowa[i-w_num:i-1])  # 標準偏差を計算します
                    #st.write(std_dev3)
                    if std_devs3:
                        last_std_dev3 = std_devs3[-1]
                        std_devs3.append((last_std_dev3 + std_dev3)*0.5)
                    else:
                        std_devs3.append(std_dev3)  # 最初の場合はそのまま追加
            #st.write(std_devs3)
            
            # データの作成
            data11 = go.Scatter(x=x+1, y=std_devs3, mode='lines+markers', line=dict(color="#77AF9C"),  name='STD3'.format(i))
            # レイアウトの作成
            layout11 = go.Layout(
                title='合算STD',
                xaxis=dict(title='work'),
                yaxis=dict(title='std'),
                showlegend=True
            )

            if st.button("解析波形出力", help="指定針目の指定区間総和値をワーク毎に表示", type='primary'):
                # フィギュアの作成
                fig7 = go.Figure(data=data8, layout=layout8)
                # フィギュアの表示
                st.plotly_chart(fig7, use_container_width=True)

                # フィギュアの表示
                st.plotly_chart(hist, use_container_width=True)
                st.write(f'歪度：{skew}（正：左寄り、負：右寄り、0に近いほど正規性あり）')
                st.write(f'尖度：{kurt}（正：尖っている、負：尖ってない、0に近いほど正規性あり）')

                # フィギュアの作成
                fig8 = go.Figure(data=data9, layout=layout9)
                # フィギュアの表示
                st.plotly_chart(fig8, use_container_width=True)

                # フィギュアの作成
                fig9 = go.Figure(data=data10, layout=layout10)
                # フィギュアの表示
                st.plotly_chart(fig9, use_container_width=True)

                # フィギュアの作成
                fig10 = go.Figure(data=data11, layout=layout11)
                # フィギュアの表示
                st.plotly_chart(fig10, use_container_width=True)
                
            else:
                st.write('解析波形出力後に「解析波形出力」をクリック!!')
            
            st.write("### ▼csv出力")
            csvout(ndsowa, '総和張力のcsv出力', 'sowa')
            csvout(std_devs1, '移動stdのcsv出力', 'i_std')
            csvout(std_devs2, '10ワークずつ増やすSTDのcsv出力', '10_std')
            csvout(std_devs3, '合算stdのcsv出力', 'g_std')
            
        else:
            st.write('### データをアップロードしてください')
            
    with tab7:
        if uploaded_files:
            st.header("SDD LOG解析", divider='red')
            st.write("### ▼σ確認")
            st.write("解析する張力総和区間を設定してください")
            col1, col2 = st.columns(2)
            with col1:
                w_st0 = st.number_input('区間1-開始位相', value=ten_st)
                st.markdown(f"<span style='font-size: 12px;'>換算</span> {w_st0 * 1.40625}°", unsafe_allow_html=True)  #カウント数を角度に換算
                w_st1 = st.number_input('区間2-開始位相', value=km_st)
                st.markdown(f"<span style='font-size: 12px;'>換算</span> {w_st1 * 1.40625}°", unsafe_allow_html=True)  #カウント数を角度に換算
                w_st2 = st.number_input('区間3-開始位相', value=0)
                st.markdown(f"<span style='font-size: 12px;'>換算</span> {w_st2 * 1.40625}°", unsafe_allow_html=True)  #カウント数を角度に換算
            with col2:
                w_en0 = st.number_input('区間1-終了位相', value=ten_ed)
                st.markdown(f"<span style='font-size: 12px;'>換算</span> {w_en0 * 1.40625}°", unsafe_allow_html=True)
                w_en1 = st.number_input('区間2-終了位相', value=km_ed)
                st.markdown(f"<span style='font-size: 12px;'>換算</span> {w_en1 * 1.40625}°", unsafe_allow_html=True)
                w_en2 = st.number_input('区間3-終了位相', value=0)
                st.markdown(f"<span style='font-size: 12px;'>換算</span> {w_en2 * 1.40625}°", unsafe_allow_html=True)
                
            st.write("出会い張力総和区間を設定してください")
            col3, col4 = st.columns(2)
            with col3:
                w_dstt = st.number_input('出会い開始位相,', value=d_st)
                st.markdown(f"<span style='font-size: 12px;'>換算</span> {w_dstt * 1.40625}°", unsafe_allow_html=True)
    
            with col4:
                w_dend = st.number_input('出会い終了位相,', value=d_ed)
                st.markdown(f"<span style='font-size: 12px;'>換算</span> {w_dend * 1.40625}°", unsafe_allow_html=True)
            
            w_stt = [w_st0, w_st1, w_st2]
            w_end = [w_en0, w_en1, w_en2]
            x = np.array(range(len(n_array)))  # 横軸針数
            nwsowa = np.zeros((len(w_array), len(n_array), 3))  # ワークx針数の配列を作成
            std = np.zeros((3, len(n_array)))
            for i in range(3):
                for w in range(len(w_array)):
                    for nd in range(len(n_array)):  # 指定区間の総和値をsowa配列に保存する
                        nwsowa[w, nd, i] = np.sum(w_array[w,nd,w_stt[i]:w_end[i],6]) - np.sum(w_array[w,nd,w_dstt:w_dend,6])*dd  # 出会い区間128:156
                tsowa = nwsowa[:, :, i]
                std[i,:] = calculate_std_deviation(tsowa)

            # 表データを作成
            data = {'針数': x+1,
                    'σ-区間1': std[0],
                    'σ-区間2': std[1],
                    'σ-区間3': std[2]}
            df = pd.DataFrame(data)
            st.table(df)

            st.write("### ▼csv出力")
            col5, col6 = st.columns(2)
            with col5:
                start_w = st.number_input('何ワーク目から出力しますか？', min_value=1, max_value=len(w_array) , step=1)
            with col6:
                end_w = st.number_input('何ワーク目まで出力しますか？', min_value=1, max_value=len(w_array) , step=1)
            csvouts(nwsowa[start_w-1:end_w,:,0], '複数ワークの総和張力のcsv出力', 'wsowa')
            
        else:
            st.write('### データをアップロードしてください')
        
    with tab8:
        if uploaded_files:
            st.header("SDD LOG解析", divider='red')
            st.write("### ▼3Dグラフ")
            # 10iワーク毎の標準偏差
            col1, col2 = st.columns(2)
            with col1:
                s_stt = st.number_input('.開始位相', value=ten_st)
                st.markdown(f"<span style='font-size: 12px;'>換算</span> {s_stt * 1.40625}°", unsafe_allow_html=True)  #カウント数を角度に換算
            with col2:
                s_end = st.number_input('.終了位相', value=ten_ed)
                st.markdown(f"<span style='font-size: 12px;'>換算</span> {s_end * 1.40625}°", unsafe_allow_html=True)
            st.write("出会い張力総和区間を設定してください")
            col3, col4 = st.columns(2)
            with col3:
                s_dstt = st.number_input('.出会い開始位相', value=d_st)
                st.markdown(f"<span style='font-size: 12px;'>換算</span> {s_dstt * 1.40625}°", unsafe_allow_html=True)
            with col4:
                s_dend = st.number_input('.出会い終了位相', value=d_ed)
                st.markdown(f"<span style='font-size: 12px;'>換算</span> {s_dend * 1.40625}°", unsafe_allow_html=True)
            st.write(f"表示するワーク数を設定してください。現在アップロード数：{len(w_array)}ワーク")
            col5, col6 = st.columns(2)
            with col5:
                n_s = st.number_input('開始ワーク数', value=1, min_value=1, max_value=len(w_array)-1 , step=1)
            with col6:
                n_e = st.number_input('終了ワーク数', value=2, min_value=2, max_value=len(w_array) , step=1)
            st.write(f"表示する針数を設定してください。最大針数：{len(n_array)}針")
            col7, col8 = st.columns(2)
            with col7:
                nd_s = st.number_input('開始針数', value=1, min_value=1, max_value=len(n_array)-1 , step=1)
            with col8:
                nd_e = st.number_input('終了針数', value=2, min_value=2, max_value=len(n_array) , step=1)
            wn = st.number_input('ここで指定したワーク毎に標準偏差を計算する', value=1)
            #xs = np.array(range(len(w_array)))
            xsowa = []
            ysowa = []
            zsowa = []
            wsowa = np.zeros((len(n_array),len(w_array)))
            for n in range(n_s-1,n_e-1):
                for nd in range(nd_s-1,nd_e-1):
                    xsowa.append(n)
                    ysowa.append(nd)
                    wsowa[nd][n] = np.sum(w_array[n,nd,s_stt:s_end,6]) - np.sum(w_array[n,nd,s_dstt:s_dend,6])*dd
                    zsowa.append(wsowa[nd][n])

            st.divider()
            # 任意の閾値
            col9, col10 = st.columns(2)
            with col9:
                th_sowa_u = st.number_input('総和張力の上限閾値を設定', value=3000)
            with col10:
                th_sowa_t = st.number_input('総和張力の加減閾値を設定', value=6000)
            # 色を変更する条件を指定
            color_sowa = ["#77AF9C" if th_sowa_u <= z_val <= th_sowa_t else 'red' for z_val in zsowa]

            # データの作成
            data12 = go.Scatter3d(
                x=xsowa,
                y=ysowa,
                z=zsowa,
                mode='markers',
                marker=dict(color=color_sowa,size=2),
                showlegend=True)
            # レイアウトの作成
            layout12 = go.Layout(
                title='総和張力',
                scene = dict(
                    xaxis_title='work',
                    yaxis_title='needle',
                    zaxis_title='Tension'
                    ))
            # フィギュアの作成
            fig11 = go.Figure(data=data12, layout=layout12)
            # フィギュアの表示
            st.plotly_chart(fig11, use_container_width=True)
            
            xstd = []
            ystd = []
            zstd = []
            wstd = np.arange(len(n_array))
            std_10 = np.zeros((len(n_array),len(w_array)//10))
            for p in range(n_s-1,n_e-1,wn):
                for q in range(nd_s-1,nd_e-1):
                    xstd.append(p)
                    ystd.append(q)
                    zstd.append(np.std(wsowa[q][0:p]))
            
            # 任意の閾値
            col11, col12 = st.columns(2)
            with col11:
                th_sigma_u = st.number_input('総和張力の上限閾値を設定', value=100)
            with col12:
                th_sigma_t = st.number_input('総和張力の加減閾値を設定', value=1000)
            # 色を変更する条件を指定
            color_sigma = ['#87ceeb' if th_sigma_u <= z_val < th_sigma_t else 'red' for z_val in zstd]

            # データの作成
            data13 = go.Scatter3d(
                x=xstd,
                y=ystd,
                z=zstd,
                mode='markers',
                marker=dict(color=color_sigma,size=2),
                showlegend=True)
            # レイアウトの作成
            layout13 = go.Layout(
                title=f'{wn}ワーク毎の標準偏差',
                scene = dict(
                    xaxis_title='(n)work',
                    yaxis_title='needle',
                    zaxis_title='STD'
                    ))
            
            # フィギュアの作成
            fig12 = go.Figure(data=data13, layout=layout13)
            # フィギュアの表示
            st.plotly_chart(fig12, use_container_width=True)
            
            

