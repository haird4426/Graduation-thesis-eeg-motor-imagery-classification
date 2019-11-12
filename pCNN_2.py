#%%
#データ増強の順番の変更
#脳波前処理後にデータ増強をして場合精度が低い

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
subject_number = 1  #被験者数
subject        = 1  #被験者を指定

stride      = 0.125 #ストライド(s)
time_window = 4     #時間窓
threshold   = 2     #ノイズ除去の閾値
#----------------------------------------------------------
each_data_number = 72               #各runファイルの試行回数
sampling_freq = 256                 #サンプリング周波数
ch_number = 3                       #チャンネル数
file_number = 3                     #各被験者ディレクトリのファイル数
imagination_time = 4                #運動想起時間
label_number = 3                    #ラベル数
delete_label = 3 - 1                #削除するラベル-1
start_time = 1.5                    #抽出開始時刻
start_imagination_time = 4.5        #運動想起開始時刻
end_time = 8.5                      #抽出終了時刻
extraction_time = end_time - start_time #抽出時間

extraction_freq = int(sampling_freq * extraction_time) #抽出時間→周波数変換
all_sample_number = int((each_data_number * 2 / 3) * file_number * subject_number)
print("asn:",all_sample_number)
#-------------------------------------------------
#print(os.getcwd()) #カレントディレクトリ
#相対パスで全ファイルにアクセス
current_path = os.getcwd() #これでPCによらない
#print("current_path:",current_path)
#-------------------------------------------------

data_x = np.zeros((ch_number,all_sample_number,extraction_freq),dtype="float64")
data_y = np.zeros(all_sample_number,dtype="int32")

run_cnt = 0
for i in range(subject_number): #各被験者ごとのループ
    #被験者を指定
    i = subject
    print("s"+str(subject))

    #毎回各被験者のデータを読み込む
    data_run1 = (scipy.io.loadmat(current_path + "/s" + str(i) + "/" + "/Run1.mat"))
    data_run2 = (scipy.io.loadmat(current_path + "/s" + str(i) + "/" + "/Run2.mat"))
    data_run3 = (scipy.io.loadmat(current_path + "/s" + str(i) + "/" + "/Run3.mat"))

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

        front_freq = int((start_imagination_time - start_time) * sampling_freq)
        rear_freq = int((end_time - start_imagination_time) * sampling_freq)
        start_imagination_freq = int(start_imagination_time * sampling_freq)

        #print(front_freq) #768
        #print(rear_freq) #1024
        #各ファイルごと読み込む
        #run1ファイルの読み込み
        if label_jdg[0][delete_label] != 1:
            data_y[run_cnt+label] = data_run1["Y"][0][j]
            for l in range(ch_number):
                for k in range(front_freq,0,-1):
                    data_x[l][run_cnt+label][front_freq-k] = data_run1["X"][starting_imagination_label[0]-k][l]
            for l in range(ch_number):
                for k in range(rear_freq):
                    data_x[l][run_cnt+label][front_freq+k] = data_run1["X"][starting_imagination_label[0]+k][l]
            label += 1

        if label_jdg[1][delete_label] != 1:
            data_y[run_cnt+label] = data_run2["Y"][0][j]
            for k in range(front_freq,0,-1):
                for l in range(ch_number):
                    data_x[l][run_cnt+label][front_freq-k] = data_run2["X"][starting_imagination_label[1]-k][l]
            for k in range(rear_freq):
                for l in range(ch_number):
                    data_x[l][run_cnt+label][front_freq+k] = data_run2["X"][starting_imagination_label[1]+k][l]
            label += 1

        if label_jdg[2][delete_label] != 1:
            data_y[run_cnt+label] = data_run3["Y"][0][j]
            for k in range(front_freq,0,-1):
                for l in range(ch_number):
                    data_x[l][run_cnt+label][front_freq-k] = data_run3["X"][starting_imagination_label[2]-k][l]
            for k in range(rear_freq):
                for l in range(ch_number):
                    data_x[l][run_cnt+label][front_freq+k] = data_run3["X"][starting_imagination_label[2]+k][l]
            label += 1
    run_cnt += int(each_data_number * 3 * 2 / 3)

#"""
#グラフ表示
x = np.linspace(0,extraction_time,extraction_freq)
import random
random_label = []
for i in range(4):
    random_label.append(random.randint(0,all_sample_number))
    y = data_x[0][random_label[i]] #ランダムに表示
    #y = data_x[0][0]
    plt.plot(x,y)
    print("label",random_label[i])
    plt.show()
#"""
#"""
#0の要素があるかチェック➡OK:ファイルの名前を変更して対処
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
#----脳波の前処理
import scipy.signal as signal
from scipy.fftpack import fft
filter = [0] * ch_number
nyquist_freq = sampling_freq * 0.5
tap_number_1 = 61
tap_number_2 = 61
tap_number_3 = 61
cutoff_freq = 0.5

N = extraction_freq
dt = 1 / sampling_freq * 2
freq = np.linspace(0, 1.0/dt, N) # frequency step
yf_1 = fft(data_x[0][0])/(N/2)

#"""
#50Hzノッチフィルター
print("50Hz notch filter")
for i in range(all_sample_number):
    #filter = signal.firwin(numtaps=tap_number_1,pass_zero=False,cutoff=[49,51],fs=sampling_freq)
    #filter = signal.firwin(numtaps=tap_number_1,nyq=nyquist_freq,pass_zero=False,cutoff=[49,51])
    filter = signal.firwin(numtaps=tap_number_1,nyq=nyquist_freq,cutoff=[49,51])

    data_x[0][i] = signal.lfilter(filter,1,data_x[0][i])
    data_x[1][i] = signal.lfilter(filter,1,data_x[1][i])
    data_x[2][i] = signal.lfilter(filter,1,data_x[2][i])
#"""
#"""
#カットオフ周波数0.5Hzのハイパスフィルター
print("High-pass filter with a cutoff frequency of 0.5Hz")
for i in range(all_sample_number):
    #filter = signal.firwin(numtaps=tap_number_2,pass_zero=False,cutoff=0.5,fs=sampling_freq)
    #filter = signal.firwin(numtaps=tap_number_2,nyq=nyquist_freq,pass_zero=False,cutoff=cutoff_freq)
    filter = signal.firwin(numtaps=tap_number_2,nyq=nyquist_freq,cutoff=cutoff_freq)

    data_x[0][i] = signal.lfilter(filter,1,data_x[0][i])
    data_x[1][i] = signal.lfilter(filter,1,data_x[1][i])
    data_x[2][i] = signal.lfilter(filter,1,data_x[2][i])
#"""
#"""
#5次の0位相バターワースフィルターで2~60Hzのバンドパスフィルター
print("2-60Hz band pass filter")
#------------------------------------バターワースフィルターを要実装
for i in range(all_sample_number):
    #filter = signal.firwin(numtaps=tap_number_3,pass_zero=False,cutoff=[2,60],fs=sampling_freq)
    #filter = signal.firwin(numtaps=tap_number_3,nyq=nyquist_freq,pass_zero=False,cutoff=[2,60])
    filter = signal.firwin(numtaps=tap_number_3,nyq=nyquist_freq,cutoff=[2,60])

    data_x[0][i] = signal.lfilter(filter,1,data_x[0][i])
    data_x[1][i] = signal.lfilter(filter,1,data_x[1][i])
    data_x[2][i] = signal.lfilter(filter,1,data_x[2][i])
#"""
"""
yf_2 = fft(data_x[0][0])/(N/2)

plt.plot(freq,np.abs(yf_1))
plt.show()
plt.plot(freq,np.abs(yf_2))
plt.show()
"""
#μ+-6σの修正外れ値にクリップ

#"""
#各チャンネルで(xi-μi)/σiで正規化
print("Normalized by (xi-μ) / σi for each channel")
avg_ch1 = np.average(data_x[0])
avg_ch2 = np.average(data_x[1])
avg_ch3 = np.average(data_x[2])

std_ch1 = np.std(data_x[0])
std_ch2 = np.std(data_x[1])
std_ch3 = np.std(data_x[2])

data_x[0] = (data_x[0] - avg_ch1) / std_ch1
data_x[1] = (data_x[1] - avg_ch2) / std_ch2
data_x[2] = (data_x[2] - avg_ch3) / std_ch3
#"""

#-----------------------------------------------------------------------------
#i番目のデータを削除する関数
def delete_data(i,all_sample_number,data_x,data_y):
    #print("old:",data_ch1.shape)
    data_x = np.delete(data_x,i,0)

    data_y = np.delete(data_y,i,0)

    #print("new:",data_ch1.shape)
    #print("label:",data_y.shape)
    all_sample_number -= 1
    i -= 1
    #print("asn:",all_sample_number)
    #print("i_2:",i)
    return i,all_sample_number,data_x,data_y
#-----------------------------------------------------------------------------
#"""

#ノイズ除去のためにreshape
data_x = data_x.reshape(all_sample_number,ch_number,extraction_freq)

#EOG,EMGアーティファクト除去
print("EOG,EMG artifact removal")
print("asn:",all_sample_number)
tmp = np.zeros(ch_number,dtype="float32")

##max<(x_ne) - min(x_ne)>_N
print("< max(x_ne) - min(x_ne) >_N")
i = 0
while i < all_sample_number:
    tmp[0] = np.amax(data_x[i][0]) - np.min(data_x[i][0])
    tmp[1] = np.amax(data_x[i][1]) - np.min(data_x[i][1])
    tmp[2] = np.amax(data_x[i][2]) - np.min(data_x[i][2])

    z = np.average(tmp)
    if z > 3:
        i,all_sample_number,data_x,data_y = delete_data(i,all_sample_number,data_x,data_y)
    i = i + 1
print("asn:",all_sample_number)

##<x_ne> - <x_n>
print("<x_ne> - <x_n>")
avg_ch_epoch = np.zeros(ch_number,dtype="float32")
avg_ch       = np.zeros(ch_number,dtype="float32")
i = 0
while i < all_sample_number:
    avg_ch_epoch[0] = np.average(data_x[i][0])
    avg_ch_epoch[1] = np.average(data_x[i][1])
    avg_ch_epoch[2] = np.average(data_x[i][2])

    avg_ch[0] = np.average(data_x[i][0])
    avg_ch[1] = np.average(data_x[i][1])
    avg_ch[2] = np.average(data_x[i][2])

    tmp[0] = np.abs(avg_ch_epoch[0] - avg_ch[0])
    tmp[1] = np.abs(avg_ch_epoch[1] - avg_ch[1])
    tmp[2] = np.abs(avg_ch_epoch[2] - avg_ch[2])

    if tmp[0] > threshold or tmp[1] > threshold or tmp[2] > threshold:
        i,all_sample_number,data_x,data_y = delete_data(i,all_sample_number,data_x,data_y)

    i += 1
print("asn:",all_sample_number)

##S^2_x_ne
print("S^2_x_ne")
i = 0
while i < all_sample_number:
    tmp[0] = np.var(data_x[i][0])
    tmp[1] = np.var(data_x[i][1])
    tmp[2] = np.var(data_x[i][2])

    if tmp[0] > threshold or tmp[1] > threshold or tmp[2] > threshold:
        i,all_sample_number,data_x,data_y = delete_data(i,all_sample_number,data_x,data_y)

    i += 1
print("asn:",all_sample_number)

##median(d(x_ne) / dt)
print("median(d(x_ne) / dt)")
dt = float(1 / sampling_freq)
i = 0
while i < all_sample_number:
    tmp[0] = np.median(np.diff(data_x[i][0]) / dt)
    tmp[1] = np.median(np.diff(data_x[i][1]) / dt)
    tmp[2] = np.median(np.diff(data_x[i][2]) / dt)
    if tmp[0] > threshold or tmp[1] > threshold or tmp[2] > threshold or tmp[0] < -threshold or tmp[1] < -threshold or tmp[2] < -threshold:
        i,all_sample_number,data_x,data_y = delete_data(i,all_sample_number,data_x,data_y)

    i += 1
print("asn:",all_sample_number)

data_x = data_x.reshape(ch_number,all_sample_number,extraction_freq)

print(data_x.shape)


#"""
#グラフ描画
random_label = []
for i in range(4):
    random_label.append(random.randint(0,all_sample_number))
    y = data_x[0][random_label[i]] #ランダムに表示
    plt.plot(x,y)
    print("label",random_label[i])
    plt.show()
#"""
#"""
#0の要素があるかチェック➡OK:ファイルの名前を変更して対処
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
#データ増強
data_magnification = int(((end_time - time_window - start_time) / stride) + 1) #データ増加率
print("data_mag:",data_magnification)

new_all_sample_number = all_sample_number * data_magnification

new_data_x = np.zeros((ch_number,new_all_sample_number,int(time_window*sampling_freq)),dtype="float64")
new_data_y = np.zeros((new_all_sample_number),dtype="int32")
#a = [stride * x for x in range(data_magnification-1,-1,-1)]
#print(a)
new_tail = int(sampling_freq * time_window - 1)
label_reference = 0
new_extraction_freq = int(sampling_freq * time_window)

"""
#直接参照して代入
for i in range(all_sample_number):
    for j in range(data_magnification):
        for k in range(ch_number):
            head = int(sampling_freq * stride * j)
            for l in range(new_extraction_freq):
                new_data_x[k][i][l] = data_x[k][i][head]
                head += 1
        new_data_y[label_reference] = data_y[i]
        label_reference += 1
"""
#スライスで代入
for i in range(all_sample_number):
    head = 0
    tail = int(sampling_freq * time_window)
    for j in range(data_magnification):
        for ch in range(ch_number):
            #new_data_x[k][i][:new_tail] = data_x[k][i][head:tail]
            new_data_x[ch][label_reference] = data_x[ch][i][head:tail]
            #print(data_x[k][i][head:tail])
            #print(new_data_x[k][i][0:new_tail])
        #print(j)
        new_data_y[label_reference] = data_y[i]
        head += int(sampling_freq * stride)
        tail += int(sampling_freq * stride)
        label_reference += 1



all_sample_number = all_sample_number * data_magnification



#print("new_data_x.shape:",new_data_x.shape)
#"""
#グラフ描画
x = np.linspace(0,time_window,sampling_freq*time_window)
random_label = []
for i in range(4):
    random_label.append(random.randint(0,all_sample_number))
    y = new_data_x[0][random_label[i]] #ランダムに表示
    #y = new_data_x[0][i]
    plt.plot(x,y)
    print("label",random_label[i])
    plt.show()
#"""
#"""
#0の要素があるかチェック
print("-----")
a = sampling_freq * time_window
b = 0
for label in range(all_sample_number):
    for j in range(a):
        if new_data_x[0][label][j] == 0:
            print("input_label:",label)
            b = 1
            break
    if b:
        break
    if new_data_y[label] == 0:
        print("output_label:",label)
#"""



#%%
#----STFT

#import scipy.io.wavfile as wio

#----------------------------ハイパーパラメータ
over_lap_number = 112
nperseg_number  = 128

#----------------------------
#論文のpCNNの値
over_lap_number = 112
nperseg_number  = 128
#---------------------------

#--------------------------------input_dataの型を計算で求める
input_data  = np.zeros((ch_number,all_sample_number,65,65),dtype="complex64")

for i in range(all_sample_number):
    f, t, input_data[0][i] = scipy.signal.stft(new_data_x[0][i],fs=sampling_freq,nperseg=nperseg_number,noverlap=over_lap_number)
    f, t, input_data[1][i] = scipy.signal.stft(new_data_x[1][i],fs=sampling_freq,nperseg=nperseg_number,noverlap=over_lap_number)
    f, t, input_data[2][i] = scipy.signal.stft(new_data_x[2][i],fs=sampling_freq,nperseg=nperseg_number,noverlap=over_lap_number)

#print("len(t):",len(t))
#print("len(f:)",len(f))

input_data = abs(input_data)
#print(scipy.signal.stft(data_ch1[0],fs=sampling_freq,noverlap=over_lap_number)[2])
#print(scipy.signal.stft(data_ch1[0],fs=sampling_freq,noverlap=over_lap_number)[2].shape)

#b = abs(scipy.signal.stft(data_ch1[0],fs=sampling_freq,noverlap=over_lap_number)[2])
#print(b)

print("input_data.shape:",input_data.shape)

"""
#各チャンネルで(xi-μi)/σiで正規化
avg_ch1 = np.average(input_data[0])
avg_ch2 = np.average(input_data[1])
avg_ch3 = np.average(input_data[2])

std_ch1 = np.std(input_data[0])
std_ch2 = np.std(input_data[1])
std_ch3 = np.std(input_data[2])

input_data[0] = (input_data[0] - avg_ch1) / std_ch1
input_data[1] = (input_data[1] - avg_ch2) / std_ch2
input_data[2] = (input_data[2] - avg_ch3) / std_ch3
"""
#print(input_data[0][0])

input_data = input_data.reshape(all_sample_number,len(f),len(t),ch_number)



#%%
#----データ整理
from sklearn import model_selection
from keras.utils import np_utils

classes = 2
test_rate = 0.1

#正解ラベルYを(each_data_number,)へ
print(new_data_y.shape)
print(input_data.shape)
#data_y = data_y.reshape(each_data_number*3*subject_number)
#print(data_y.shape) #(each_data_number,)
#print(input_data.shape)
#input_data = input_data.reshape(each_data_number,129*19*4)
#print(input_data.shape)

#出力Yのラベルを1,2,3から0,1,2に変更
for i in range(len(new_data_y)):
    if new_data_y[i] == 1:
        new_data_y[i] = 0
    elif new_data_y[i] == 2:
        new_data_y[i] = 1
    elif new_data_y[i] == 3:
        new_data_y[i] = 2
#print(data_y) #OK

#正解ラベルYをont-hot表現へ
new_data_y = np_utils.to_categorical(new_data_y,classes)
#print(data_y) #OK

#以下にモデルの検証の仕方
#https://newtechnologylifestyle.net/%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%80%81%E3%83%87%E3%82%A3%E3%83%BC%E3%83%97%E3%83%A9%E3%83%BC%E3%83%8B%E3%83%B3%E3%82%B0%E3%81%A7%E3%81%AE%E5%AD%A6%E7%BF%92%E3%83%87%E3%83%BC%E3%82%BF%E3%81%A8/

####データの分割####
x_train, x_test, y_train, y_test = model_selection.train_test_split(input_data, new_data_y, test_size=test_rate)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)



#%%
#pCNNモデル構築
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Activation
from keras.optimizers import Adam,RMSprop,Adagrad,Adadelta,Adamax,Nadam
from keras.layers import BatchNormalization,Dropout
from keras.initializers import RandomNormal

optimizer = Adam(lr=0.00001)
epoch = 50
dropout_rate = 0.2
strides = None #default None
pool_size = (2,2)

first_filter_number  = 100  #24
second_filter_number = 200  #48
third_filter_number  = 400  #96
fourth_filter_number = 192

first_kernel_size  = 12 #12
second_kernel_size = 8  #8
third_kernel_size  = 4  #4
fourth_kernel_size = 2
activation_function = "relu"

model = Sequential()
#第１層目
model.add(Conv2D(filters=first_filter_number,
            kernel_size=first_kernel_size,
            padding="same",
            bias_initializer="he_normal",
            kernel_initializer="he_normal",
            input_shape=(len(f),len(t),ch_number)))
#"""
model.add(Conv2D(filters=first_filter_number,
            kernel_size=first_kernel_size,
            padding="same",
            activation=activation_function,
            bias_initializer="he_normal",
            kernel_initializer="he_normal"))
#"""
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=pool_size,strides=strides))
model.add(Activation(activation_function))

#第2層目
model.add(Conv2D(filters=second_filter_number,
            kernel_size=second_kernel_size,
            padding="same",
            bias_initializer="he_normal",
            kernel_initializer="he_normal",
            ))
#"""
model.add(Conv2D(filters=second_filter_number,
            kernel_size=second_kernel_size,
            padding="same",
            activation=activation_function,
            bias_initializer="he_normal",
            kernel_initializer="he_normal"))
#"""
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=pool_size,strides=strides))
model.add(Activation(activation_function))
#"""
#第3層目
model.add(Conv2D(filters=third_filter_number,
            kernel_size=third_kernel_size,
            padding="same",
            bias_initializer="he_normal",
            kernel_initializer="he_normal",
            ))

model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=pool_size,strides=strides))
model.add(Activation(activation_function))
#"""
#"""
#第4層目
model.add(Conv2D(filters=fourth_filter_number,
            kernel_size=fourth_kernel_size,
            padding="same",
            activation=activation_function,
            bias_initializer="he_normal",
            kernel_initializer="he_normal"))

model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=pool_size))
#"""

model.add(Dropout(dropout_rate))
model.add(Flatten())
model.add(Dense(units=classes,activation="softmax"))

model.compile(loss = "binary_crossentropy",
                optimizer = optimizer,
                metrics = ["accuracy"])
model.summary()



#%%
#----学習
#--------------------------------------------
#gpuの必要なメモリしか使わない文
import tensorflow as tf
from keras.backend import tensorflow_backend

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)
#--------------------------------------------
#earlystopping
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss',
                                patience=10,
                                verbose=1,
                                mode='auto'
                                )

history = model.fit(x_train,
                    y_train,
                    epochs=epoch,
                    batch_size=256,
                    validation_data=(x_test,y_test)
                    #,callbacks=[early_stopping]
                    )



#%%
#----グラフ描画
import matplotlib.pyplot as plt

#Accuracy
#plt.figure(facecolor="azure", edgecolor="coral", linewidth=2)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#loss
#plt.figure(facecolor="azure", edgecolor="coral", linewidth=2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


#%%
