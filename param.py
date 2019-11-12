import os
from scipy.signal import windows

#----------------------------------------------------------
subject_number   = 1       #被験者数
subject          = 1       #被験者を指定
stride           = 0.01   #ストライド(s)
time_window      = 4       #時間窓
threshold        = 3.0     #ノイズ除去の閾値
delete_label     = 3 - 1   #削除するラベル-1
#----------------------------------------------------------
each_data_number = 72      #各runファイルの試行回数
sampling_freq    = 256     #サンプリング周波数
ch_number        = 3       #チャンネル数
file_number      = 3       #各被験者ディレクトリのファイル数
imagination_time = 4       #運動想起時間
label_number     = 3       #ラベル数
start_time       = 1.5     #抽出し始める時刻
end_time         = 8.5     #抽出し終わる時刻
#----------------------------------------------------------
#相対パスで全ファイルにアクセス
current_path = os.getcwd() #これでPCによらない
#----------------------------------------------------------
data_magnification = int(((end_time - time_window - start_time) / stride) + 1) #データ増加率
extraction_freq = sampling_freq * time_window #抽出時間➡周波数変換
all_sample_number = int(each_data_number * file_number * subject_number * data_magnification * 2 / 3)

N = extraction_freq #データ数
#ハニング窓
hanning_window = windows.hann(N)
#ハミング窓
hamming_window = windows.hamming(N)
#ブラックマン窓
black_window   = windows.blackmanharris(N)

#stft parameter
input_data_height = 65
input_data_width  = 65
over_lap_number   = 112 #112
nperseg_number    = 128 #128

#cnn parameter
classes = 2     #正解クラス数
test_rate = 0.2 #テストデータ率

lr = 1e-4
epoch = 100
dropout_rate = 0
strides = None #default None
pool_size = (2,2)
regularization_parameter = 0
layer_number = 3

filter_number = [] * layer_number
kernel_size = [] * layer_number

filter_number.append(24) #24
filter_number.append(48) #48
filter_number.append(96) #96
filter_number.append(200)
filter_number.append(200)

kernel_size.append(12) #12
kernel_size.append(8) #8
kernel_size.append(4) #4
kernel_size.append(2)
kernel_size.append(2)

activation_function = "relu"