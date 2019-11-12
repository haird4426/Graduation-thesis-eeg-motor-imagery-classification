#%%
#pCNN
#データ増強を先にやっている
#データ読み込みを改良➡高速化
#unityへデータを送るのも実装
#前処理を自作で実装➡レイヤー数3,stride=0.1で0.9突破

#----データセットの整形
import scipy.io
import numpy as np
from matplotlib import pyplot as plt
import os

#--------------------------------------------------------------
#tsとttsの最も近い値を求めるときに使用
def getNearestValue(list, num):
    """
    概要: リストからある値に最も近い値を返却する関数
    @param list: データ配列
    @param num: 対象値
    @return 対象値に最も近い値
    """

    # リスト要素と対象値の差分を計算し最小値のインデックスを取得
    idx = np.abs(np.asarray(list) - num,dtype="float64").argmin()
    return list[idx]
#--------------------------------------------------------------
#----------------------------------------------------------
subject_number = 1        #被験者数
subject        = 1        #被験者を指定

stride = 1             #ストライド(s)
time_window = 4           #時間窓
threshold = 3.0             #ノイズ除去の閾値
delete_label     = 3 - 1  #削除するラベル-1
#----------------------------------------------------------
each_data_number = 72     #各runファイルの試行回数
sampling_freq    = 256    #サンプリング周波数
ch_number        = 3      #チャンネル数
file_number      = 3      #各被験者ディレクトリのファイル数
imagination_time = 4      #運動想起時間
label_number     = 3      #ラベル数
start_time       = 1.5    #抽出し始める時刻
end_time         = 8.5    #抽出し終わる時刻

data_magnification = int(((end_time - time_window - start_time) / stride) + 1) #データ増加率
print("data_mag:",data_magnification)

extraction_freq = sampling_freq * time_window #抽出時間➡周波数変換
#stride_freq = int(sampling_freq * stride) #ストライド時間→周波数変換
all_sample_number = int(each_data_number * file_number * subject_number * data_magnification * 2 / 3)
#print("asn:",all_sample_number)
#-------------------------------------------------
#print(os.getcwd()) #カレントディレクトリ
#相対パスで全ファイルにアクセス
current_path = os.getcwd() #これでPCによらない
#print("current_path:",current_path)
#-------------------------------------------------

data_x = np.zeros((ch_number,all_sample_number,extraction_freq),dtype="float64")
data_y = np.zeros(all_sample_number,dtype="int32")
print("data_x.shape",data_x.shape)
run_cnt = 0
for i in range(subject_number): #各被験者ごとのループ
    #被験者を指定
    sub = i + 1
    sub = subject
    print("s"+str(sub))

    #毎回各被験者のデータを読み込む
    data_run1 = (scipy.io.loadmat(current_path + "/s" + str(sub) + "/" + "/Run1.mat"))
    data_run2 = (scipy.io.loadmat(current_path + "/s" + str(sub) + "/" + "/Run2.mat"))
    data_run3 = (scipy.io.loadmat(current_path + "/s" + str(sub) + "/" + "/Run3.mat"))

    #ttsとtsの型を変更
    data_run1["trial_time_stamps"] = data_run1["trial_time_stamps"].reshape(each_data_number)
    data_run2["trial_time_stamps"] = data_run2["trial_time_stamps"].reshape(each_data_number)
    data_run3["trial_time_stamps"] = data_run3["trial_time_stamps"].reshape(each_data_number)

    data_run1["time_stamps"] = data_run1["time_stamps"].reshape(data_run1["time_stamps"].size)
    data_run2["time_stamps"] = data_run2["time_stamps"].reshape(data_run2["time_stamps"].size)
    data_run3["time_stamps"] = data_run3["time_stamps"].reshape(data_run3["time_stamps"].size)

    starting_imagination_label = np.zeros(file_number,dtype="int32")
    starting_imagination_time  = np.zeros(file_number,dtype="float64")
    label = 0
    for j in range(each_data_number): #ラベル数のループ回数→すべてのtts分を見る
        label_jdg = np.zeros((file_number,label_number),dtype="int32")
        #必要に応じて削除するラベルを決める
        #run1ラベル1か否か
        if data_run1["Y"][0][j] == 1:
            label_jdg[0][0] = 1
        #run1ラベル2か否か
        if data_run1["Y"][0][j] == 2:
            label_jdg[0][1] = 1
        #run1ラベル3か否か
        if data_run1["Y"][0][j] == 3:
            label_jdg[0][2] = 1

        #run2ラベル1か否か
        if data_run2["Y"][0][j] == 1:
            label_jdg[1][0] = 1
        #run2ラベル2か否か
        if data_run2["Y"][0][j] == 2:
            label_jdg[1][1] = 1
        #run2ラベル3か否か
        if data_run2["Y"][0][j] == 3:
            label_jdg[1][2] = 1

        #run3ラベル1か否か
        if data_run3["Y"][0][j] == 1:
            label_jdg[2][0] = 1
        #run3ラベル2か否か
        if data_run3["Y"][0][j] == 2:
            label_jdg[2][1] = 1
        #run3ラベル3か否か
        if data_run3["Y"][0][j] == 3:
            label_jdg[2][2] = 1

        #ttsに最も近いtsを見つける
        starting_imagination_time[0]  = getNearestValue(data_run1["time_stamps"],data_run1["trial_time_stamps"][j])
        starting_imagination_time[1]  = getNearestValue(data_run2["time_stamps"],data_run2["trial_time_stamps"][j])
        starting_imagination_time[2]  = getNearestValue(data_run3["time_stamps"],data_run3["trial_time_stamps"][j])

        starting_imagination_label[0] = np.where(data_run1["time_stamps"]==getNearestValue(data_run1["time_stamps"],data_run1["trial_time_stamps"][j]))[0]
        starting_imagination_label[1] = np.where(data_run2["time_stamps"]==getNearestValue(data_run2["time_stamps"],data_run2["trial_time_stamps"][j]))[0]
        starting_imagination_label[2] = np.where(data_run3["time_stamps"]==getNearestValue(data_run3["time_stamps"],data_run3["trial_time_stamps"][j]))[0]

        #データ増強しながら取り込む
        a = [stride * x for x in range(data_magnification-1,-1,-1)]
        for k in a:
            k = int(k * sampling_freq)
            if label_jdg[0][delete_label] != 1:
                for ch in range(ch_number):
                    head = starting_imagination_label[0]-k
                    tail = starting_imagination_label[0]-k+extraction_freq
                    data_x[ch][run_cnt+label] = data_run1["X"][head:tail,ch]
                data_y[run_cnt+label] = data_run1["Y"][0][j]
                label += 1
            if label_jdg[1][delete_label] != 1:
                for ch in range(ch_number):
                    head = starting_imagination_label[1]-k
                    tail = starting_imagination_label[1]-k+extraction_freq
                    data_x[ch][run_cnt+label] = data_run2["X"][head:tail,ch]
                data_y[run_cnt+label] = data_run2["Y"][0][j]
                label += 1

            if label_jdg[2][delete_label] != 1:
                for ch in range(ch_number):
                    head = starting_imagination_label[2]-k
                    tail = starting_imagination_label[2]-k+extraction_freq
                    data_x[ch][run_cnt+label] = data_run3["X"][head:tail,ch]
                data_y[run_cnt+label] = data_run3["Y"][0][j]
                label += 1
    run_cnt += int(each_data_number * 2 * data_magnification)

#"""
#グラフ表示
x = np.linspace(0,imagination_time,extraction_freq)
#x = [0] * extraction_freq
#print(x)
import random
random_label = []
for i in range(3):
    random_label.append(random.randint(0,all_sample_number))
    y = data_x[0][random_label[i]] #ランダムに表示
    #y = data_x[0][i+3742]
    plt.plot(x,y)
    print("label",random_label[i])
    plt.show()
#"""


#%%
#----前処理
from scipy.fftpack import fft,fftfreq,ifft

fft_wave_complex = np.zeros((ch_number,all_sample_number,extraction_freq),dtype="complex128")
fft_wave_abs     = np.zeros((ch_number,all_sample_number,extraction_freq),dtype="float64")

"""
f = fft(data_x[0][0])
g = fftfreq(n=data_x[0][0].size, d=1/sampling_freq)
print(f.shape)
print(g.shape)
"""

#fft
fft_wave_complex[:][:] = fft(data_x[:][:])
fft_fre = fftfreq(n=data_x[0][0].size, d=1/sampling_freq)

#50Hz noach filter
print("50Hz notch filter")
fft_pass = np.where(np.abs(fft_fre)!=50,fft_fre,0)
fft_pass = np.where(fft_pass==0,fft_pass,1)
fft_pass[0] = 1
fft_wave_complex[:][:] *= fft_pass

"""
#0.5Hzハイパスフィルター
print("High-pass filter with a cutoff frequency of 0.5Hz")
fft_pass = np.where(np.abs(fft_fre)>0.5,fft_fre,0)
fft_pass = np.where(fft_pass==0,fft_pass,1)
fft_wave_complex[:][:] *= fft_pass
"""

#2~60Hzバンドパスフィルター
print("2-60Hz band pass filter")
fft_pass = np.where((2<=np.abs(fft_fre)) & (np.abs(fft_fre)<=60),fft_fre,0)
fft_pass = np.where(fft_pass==0,fft_pass,1)
fft_wave_complex[:][:] *= fft_pass

#ifft
#print(abs(fft_wave_complex[:][:]))
print("a")
#fft_wave_abs[:][:] = abs(fft_wave_complex[:][:])
print("b")
#fft_wave_abs = ifft(fft_wave_complex[:][:])
data_x[:][:] = ifft(fft_wave_complex[:][:])
print(data_x[0][0])
# data_x[:][:] = abs(data_x[:][:])
print("c")

#μ+-6σの修正外れ値にクリップ

#各チャンネルで(xi-μi)/σiで標準化
print("Normalized by (xi-μ) / σi for each channel")
avg = np.zeros((ch_number),dtype="float64")
std = np.zeros((ch_number),dtype="float64")
print("d")
data_x = data_x.reshape((ch_number,extraction_freq,all_sample_number))
#"""
for ch in range(ch_number):
    #for i in range(all_sample_number):
    #for i in range(extraction_freq):
        #avg[ch][i] = np.average(data_x[ch][i])
        #std[ch][i] = np.std(data_x[ch][i])
    print("ch:",ch)
    if ch == 1:
        print(data_x[ch])
    avg[ch] = np.average(data_x[ch])
    std[ch] = np.std(data_x[ch])
print("e")

for ch in range(ch_number):
    #for i in range(all_sample_number):
    #for i in range(extraction_freq):
    print(avg[ch])
    print(std[ch])
    data_x[ch] = (data_x[ch] - avg[ch]) / std[ch]
#"""
data_x = data_x.reshape((ch_number,all_sample_number,extraction_freq))
print("f")
#"""
#グラフ描画
random_label = []
for i in range(3):
    random_label.append(random.randint(0,all_sample_number))
    y = data_x[0][random_label[i]] #ランダムに表示
    plt.plot(x,y)
    print("label",random_label[i])
    plt.show()
#"""
#"""
#0の要素があるかチェック
print("-----")
for label in range(all_sample_number):
    for j in range(extraction_freq):
        if data_x[0][label][j] == 0:
            print("input_label:",label)
            break
    if data_y[label] == 0:
        print("output_label:",label)
#"""



#%%
#----STFT
from scipy.signal import stft
#----------------------------ハイパーパラメータ
over_lap_number = 110
nperseg_number  = 126
#----------------------------
#"""
#論文のpCNNの値
#over_lap_number = 112
#nperseg_number  = 128
#"""
#---------------------------

#--------------------------------input_dataの型を計算で求める
input_data  = np.zeros((ch_number,all_sample_number,64,65),dtype="complex128")

f, t, input_data[0:3,:all_sample_number] = stft(data_x[0:3,:all_sample_number],fs=sampling_freq,nperseg=nperseg_number,noverlap=over_lap_number)

print("len(f:)",len(f)) #左
print("len(t):",len(t)) #右

print(f)
print(t)

input_data = abs(input_data)

print("input_data.shape:",input_data.shape)

"""
#max-minを確認➡STFTではせいぜい2.0もない
for ch in range(ch_number):
    for asn in range(all_sample_number):
        max = np.amax(input_data[ch][asn])
        min = np.min(input_data[ch][asn])
        print("max-min",max-min)
"""
"""
#各チャンネルで正規化
for ch in range(ch_number):
    for asn in range(all_sample_number):
        max = np.amax(input_data[ch][asn])
        input_data[ch][asn] = input_data[ch][asn] / max

print(input_data[0][0])
"""

#plt.figure()
#input_data[:][:] = 10 * np.log(input_data[:][:])
plt.pcolormesh(t, f, input_data[0][0])#, cmap = "jet")
plt.ylim([f[1], f[-1]])
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.yscale('log')
plt.show()


input_data = input_data.reshape(all_sample_number,len(f),len(t),ch_number)
print(all_sample_number)


#%%
# -*- coding: utf-8 -*-
# ==================================
#
#    Short Time Fourier Trasform
#
# ==================================
from scipy import ceil, complex64, float64, hamming, zeros
from scipy.fftpack import fft# , ifft
from scipy import ifft # こっちじゃないとエラー出るときあった気がする
from scipy.io.wavfile import read

from matplotlib import pylab as pl

# ======
#  STFT
# ======
"""
x : 入力信号(モノラル)
win : 窓関数
step : シフト幅
"""
def stft(x, win, step):
    l = int(len(x)) # 入力信号の長さ
    N = int(len(win)) # 窓幅、つまり切り出す幅
    M = int(ceil(float(l - N + step) / step)) # スペクトログラムの時間フレーム数

    new_x = zeros((N + ((M - 1) * step)), dtype = float64)
    new_x[: l] = x # 信号をいい感じの長さにする

    X = zeros([M, N], dtype = complex64) # スペクトログラムの初期化(複素数型)
    for m in range(M):
        start = step * m
        X[m, :] = fft(new_x[start : start + N] * win)
    return X

# =======
#  iSTFT
# =======
def istft(X, win, step):
    M, N = X.shape
    assert (len(win) == N), "FFT length and window length are different."

    l = (M - 1) * step + N
    x = zeros(l, dtype = float64)
    wsum = zeros(l, dtype = float64)
    for m in range(M):
        start = step * m
        ### 滑らかな接続
        x[start : start + N] = x[start : start + N] + ifft(X[m, :]).real * win
        wsum[start : start + N] += win ** 2
    pos = (wsum != 0)
    x_pre = x.copy()
    ### 窓分のスケール合わせ
    x[pos] /= wsum[pos]
    return x


if __name__ == "__main__":
    fs = 256
    data = data_x[0][0]

    fftLen = 512 # とりあえず
    win = hamming(fftLen) # ハミング窓
    step = fftLen / 4

    ### STFT
    spectrogram = stft(data, win, step)

    ### iSTFT
    resyn_data = istft(spectrogram, win, step)

    ### Plot
    fig = pl.figure()
    fig.add_subplot(311)
    pl.plot(data)
    pl.xlim([0, len(data)])
    pl.title("Input signal", fontsize = 20)
    fig.add_subplot(312)
    pl.imshow(abs(spectrogram[:, : fftLen / 2 + 1].T), aspect = "auto", origin = "lower")
    pl.title("Spectrogram", fontsize = 20)
    fig.add_subplot(313)
    pl.plot(resyn_data)
    pl.xlim([0, len(resyn_data)])
    pl.title("Resynthesized signal", fontsize = 20)
    pl.show()