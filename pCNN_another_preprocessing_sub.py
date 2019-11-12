#%%
#pCNN
#pythonのライブラリによるフィルター処理
#データ増強を先にやっている
#データ読み込みを改良➡高速化
#unityへデータを送るのも実装
#前処理を自作で実装➡レイヤー数3,stride=0.1で0.9突破
#交差検証なし➡readsumよりも低く過学習する
#標準化実験

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
subject        = 18        #被験者を指定

stride = 0.02             #ストライド(s)
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
#標準化
print("Normalized by (xi-μ) / σi for each channel")
avg = np.zeros((ch_number),dtype="float64")
std = np.zeros((ch_number),dtype="float64")
###
#列185
#data_x = data_x.reshape((ch_number,extraction_freq,all_sample_number))
###
#for ch in range(ch_number):
    #for i in range(all_sample_number): #行
#    for i in range(extraction_freq):    #列
#        avg = np.average(data_x[ch][i])
#        std = np.std(data_x[ch][i])
#        data_x[ch][i] = (data_x[ch][i] - avg) / std
    #avg[ch] = np.average(data_x[ch]) #chごとの全体
    #std[ch] = np.std(data_x[ch])     #chごとの全体

#for ch in range(ch_number):
    #for i in range(all_sample_number):
    #for i in range(extraction_freq):
    #data_x[ch] = (data_x[ch] - avg[ch]) / std[ch]

#全体
#data_x = ( data_x[:][:] - np.average(data_x[:]) ) / np.std(data_x[:])
#"""
#列
#data_x = data_x.reshape((ch_number,all_sample_number,extraction_freq))
#"""
#グラフ表示
x = np.linspace(0,imagination_time,extraction_freq)
#x = [0] * extraction_freq
print(x)
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
#"""
#0の要素があるかチェック➡OK:ファイルの名前を変更して対処
print("-----")
bn = 0
for label in range(all_sample_number):
    for j in range(extraction_freq):
        if data_x[0][label][j] == 0:
            print("input_label:",label)
            bn = 1
            break
    if bn:
        break
    if data_y[label] == 0:
        print("output_label:",label)
#"""



#%%
#----前処理
import scipy.signal as signal
from scipy.fftpack import fft

filter = [0] * ch_number
tap_number = 61
nyquist_freq = sampling_freq * 0.5
"""
N = extraction_freq
dt = 1 / sampling_freq * 2
freq = np.linspace(0, 1.0/dt, N) # frequency step
yf_1 = fft(data_x[0][0])/(N/2)
"""
#"""
#50Hzノッチフィルター
print("50Hz notch filter")
filter = signal.firwin(numtaps=tap_number,cutoff=[49,51],fs=sampling_freq)
data_x[0:3,:all_sample_number] = signal.lfilter(filter,1,data_x[0:3,:all_sample_number])
"""
yf_2 = fft(data_x[0][0])/(N/2)

plt.plot(freq,np.abs(yf_1))
plt.show()
plt.plot(freq,np.abs(yf_2))
plt.show()
"""
#"""
#カットオフ周波数0.5Hzのハイパスフィルター
print("High-pass filter with a cutoff frequency of 0.5Hz")
filter = signal.firwin(numtaps=tap_number,cutoff=0.5,fs=sampling_freq)
data_x[0:3,:all_sample_number] = signal.lfilter(filter,1,data_x[0:3,:all_sample_number])

#"""
#5次の0位相バターワースフィルターで2~60Hzのバンドパスフィルター
print("2-60Hz band pass filter")
#------------------------------------バターワースフィルターを要実装
#---------------------------------------
max_freq = 30
filter = signal.firwin(numtaps=tap_number,cutoff=[2,max_freq],fs=sampling_freq)
data_x[0:3,:all_sample_number] = signal.lfilter(filter,1,data_x[0:3,:all_sample_number])
#μ+-6σの修正外れ値にクリップ
#----------------------------------------



#(ch_number,all_sample_number,extraction_freq)

#μ+-6σの修正外れ値にクリップ

#各チャンネルで(xi-μi)/σiで標準化
print("Normalized by (xi-μ) / σi for each channel")
avg = np.zeros((ch_number),dtype="float64")
std = np.zeros((ch_number),dtype="float64")
###
#列288へ
#data_x = data_x.reshape((ch_number,extraction_freq,all_sample_number))
###
#for ch in range(ch_number):
#    for i in range(all_sample_number): #行
#    for i in range(extraction_freq):    #列
#        avg = np.average(data_x[ch][i])
#        std = np.std(data_x[ch][i])
#        data_x[ch][i] = (data_x[ch][i] - avg) / std
#    avg[ch] = np.average(data_x[ch]) #chごとの全体
#    std[ch] = np.std(data_x[ch])     #chごとの全体

#for ch in range(ch_number):
    #for i in range(all_sample_number):
    #for i in range(extraction_freq):
#    data_x[ch] = (data_x[ch] - avg[ch]) / std[ch]
#print(np.average(data_x[0]))
#print(np.std(data_x[0]))

#列
#data_x = data_x.reshape((ch_number,all_sample_number,extraction_freq))

#全体
data_x = ( data_x[:] - np.average(data_x[:]) ) / np.std(data_x[:])
print(np.average(data_x))
print(np.std(data_x))

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
from scipy import signal
#----------------------------ハイパーパラメータ
over_lap_number = 128
nperseg_number  = 143
#----------------------------
#"""
#論文のpCNNの値
over_lap_number = 112
nperseg_number  = 128
#"""
#---------------------------

#--------------------------------input_dataの型を計算で求める
input_data  = np.zeros((ch_number,all_sample_number,65,65),dtype="complex128")

f, t, input_data[0:3,:all_sample_number] = scipy.signal.stft(data_x[0:3,:all_sample_number],fs=sampling_freq,nperseg=nperseg_number,noverlap=over_lap_number)

print("len(t):",len(t))
print("len(f:)",len(f))

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

input_data = input_data.reshape(all_sample_number,len(f),len(t),ch_number)
print(all_sample_number)


#%%
#----データ整理
from sklearn import model_selection
from keras.utils import np_utils

classes = 2
test_rate = 0.2

#正解ラベルYを(each_data_number,)へ
print(data_y.shape)
print(input_data.shape)

#出力Yのラベルを1,2,3から0,1,2に変更
for i in range(len(data_y)):
    if data_y[i] == 1:
        data_y[i] = 0
    elif data_y[i] == 2:
        data_y[i] = 1
    elif data_y[i] == 3:
        data_y[i] = 2
#print(data_y) #OK

#正解ラベルYをont-hot表現へ
data_y = np_utils.to_categorical(data_y,classes)
#print(data_y) #OK

#以下にモデルの検証の仕方
#https://newtechnologylifestyle.net/%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%80%81%E3%83%87%E3%82%A3%E3%83%BC%E3%83%97%E3%83%A9%E3%83%BC%E3%83%8B%E3%83%B3%E3%82%B0%E3%81%A7%E3%81%AE%E5%AD%A6%E7%BF%92%E3%83%87%E3%83%BC%E3%82%BF%E3%81%A8/

####データの分割####
x_train, x_test, y_train, y_test = model_selection.train_test_split(input_data, data_y, test_size=test_rate)

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
from keras import regularizers

optimizer = Adam(lr=1e-4)
epoch = 500
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

model = Sequential()
#第１層目
model.add(Conv2D(filters=filter_number[0],
            kernel_size=kernel_size[0],
            padding="same",
            bias_initializer="he_normal",
            kernel_initializer="he_normal",
            #bias_regularizer = regularizers.l2(regularization_parameter),
            kernel_regularizer = regularizers.l2(regularization_parameter),
            #activity_regularizer = regularizers.l2(regularization_parameter),
            input_shape=(len(f),len(t),ch_number)))

model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=pool_size,strides=strides))
model.add(Activation(activation_function))

#第n層目
for layer in range(1,int(layer_number)):
    model.add(Conv2D(filters=int(filter_number[layer]),
            kernel_size=int(kernel_size[layer]),
            padding="same",
            bias_initializer="he_normal",
            kernel_initializer="he_normal",
            #bias_regularizer = regularizers.l2(regularization_parameter),
            kernel_regularizer = regularizers.l2(regularization_parameter),
            #activity_regularizer = regularizers.l2(regularization_parameter),
    ))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=pool_size,strides=strides))
    model.add(Activation(activation_function))

model.add(Flatten())
model.add(Dropout(dropout_rate))
model.add(Dense(units=classes,activation="softmax"))

model.compile(loss = "binary_crossentropy",
                optimizer = optimizer,
                metrics = ["accuracy"])
model.summary()



#%%
#----学習
save_cnn_number = 1
#------------------------------------------------
#gpuの必要なメモリしか使わない文
import tensorflow as tf
from keras.backend import tensorflow_backend

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)
#------------------------------------------------

history = model.fit(x_train,
                    y_train,
                    epochs=epoch,
                    batch_size=256,
                    validation_data=(x_test,y_test),
                    )

#maxを保存するようにする必要ある
#モデル保存
open("cnn_"+str(save_cnn_number)+".json","w").write(model.to_json())

#学習済みの重みを保存
model.save_weights("cnn_"+str(save_cnn_number)+"_weight.h5")



#%%
#----交差検証
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
#--------------------------------------------
#gpuの必要なメモリしか使わない文
import tensorflow as tf
from keras.backend import tensorflow_backend

save_cnn_number = 1

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)
#--------------------------------------------
kf = KFold(n_splits=5, shuffle=True)
all_loss=[]
all_val_loss=[]
all_acc=[]
all_val_acc=[]
for train_index, val_index in kf.split(x_train,y_train):

    train_data=x_train[train_index]
    train_label=y_train[train_index]
    val_data=x_train[val_index]
    val_label=y_train[val_index]

    history = model.fit(train_data,
                        train_label,
                        epochs=epoch,
                        batch_size=256,
                        validation_data=(val_data,val_label),
                        )

    loss=history.history['loss']
    val_loss=history.history['val_loss']
    acc=history.history['acc']
    val_acc=history.history['val_acc']

    all_loss.extend(loss)
    all_val_loss.extend(val_loss)
    all_acc.extend(acc)
    all_val_acc.extend(val_acc)
#----------------------------------
#maxを保存するようにする必要ある
#モデル保存
open("cnn_"+str(save_cnn_number)+".json","w").write(model.to_json())

#学習済みの重みを保存
model.save_weights("cnn_"+str(save_cnn_number)+"_weight.h5")



#%%
#----グラフ描画
#loss
#plt.figure(facecolor="azure", edgecolor="coral", linewidth=2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
#plt.plot(all_loss)
#plt.plot(all_val_loss)

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#Accuracy
#plt.figure(facecolor="azure", edgecolor="coral", linewidth=2)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
#plt.plot(all_acc)
#plt.plot(all_val_acc)

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#print(max(all_val_acc))
print(max(history.history["val_acc"]))
#print(len(y_test))


#%%
#----未知データで検証
from keras.models import model_from_json

test_number = len(y_test)
model_number = 1
file_number = str(1)
predict_y = np.zeros((model_number,classes))
predict_label = np.zeros((model_number),dtype="int32")

#モデル読み込み
model = model_from_json(open("cnn_" + file_number + ".json","r").read())

#重み読み込み
model.load_weights("cnn_" + file_number + "_weight.h5")

#data_yをone-hot表現から戻す➡試験的に必要なだけ
#
#data_y = np.argmax(data_y,axis=1)
#

cnt = 0
all_cnt = 0

#読み込んだ学習済みモデルでtest
for label in range(test_number):
    #並列処理で精度が改善するか確認
    predict_y[0] = model.predict(x_test[label].reshape(1,len(f),len(t),ch_number))
    for j in range(model_number):
        if predict_y[j][0] > predict_y[j][1]:
            predict_label[j] = 0
        else:
            predict_label[j] = 1

    #多数決
    cnt_label_1 = np.count_nonzero(predict_label == 0)
    cnt_label_2 = np.count_nonzero(predict_label == 1)

    if cnt_label_1 > cnt_label_2:
        determined_label = 0
    else:
        determined_label = 1

    if determined_label == data_y[label]:
        cnt += 1
    all_cnt += 1

score = cnt / all_cnt
print("score:",score)



#%%
#----予測ラベルをunityに送信
import socket
import random
import time
from tqdm import tqdm

HOST = '127.0.0.1'
PORT = 50007

test_number = 10000
test_number = len(y_test)
model_number = 1
model = []
predict_y = np.zeros((model_number,classes))
predict_label = np.zeros((model_number),dtype="int32")

#data_yをone-hot表現から戻す➡試験的に必要なだけ
#
#data_y = np.argmax(data_y,axis=1)
#

#モデル読み込み
model_1 = model_from_json(open("cnn_"+str(1)+".json","r").read())
model_2 = model_from_json(open("cnn_"+str(2)+".json","r").read())
model_3 = model_from_json(open("cnn_"+str(3)+".json","r").read())
model_4 = model_from_json(open("cnn_"+str(4)+".json","r").read())
model_5 = model_from_json(open("cnn_"+str(5)+".json","r").read())
model_6 = model_from_json(open("cnn_"+str(6)+".json","r").read())

#重み読み込み
model_1.load_weights("cnn_"+str(1)+"_weight.h5")
model_2.load_weights("cnn_"+str(2)+"_weight.h5")
model_3.load_weights("cnn_"+str(3)+"_weight.h5")
model_4.load_weights("cnn_"+str(4)+"_weight.h5")
model_5.load_weights("cnn_"+str(5)+"_weight.h5")
model_6.load_weights("cnn_"+str(6)+"_weight.h5")

cnt = 0
all_cnt = 0

#読み込んだ学習済みモデルで予測
#while True:
for i in tqdm(range(test_number)):
    #並列処理で精度が改善するか確認
    label = random.randrange(len(y_test))
    predict_y[0] = model_6.predict(x_test[label].reshape(1,len(f),len(t),ch_number))
    """
    predict_y[0] = model_1.predict(x_test[label].reshape(1,len(f),len(t),ch_number))
    predict_y[1] = model_2.predict(x_test[label].reshape(1,len(f),len(t),ch_number))
    predict_y[2] = model_3.predict(x_test[label].reshape(1,len(f),len(t),ch_number))
    predict_y[3] = model_4.predict(x_test[label].reshape(1,len(f),len(t),ch_number))
    predict_y[4] = model_5.predict(x_test[label].reshape(1,len(f),len(t),ch_number))
    """
    """
    #print(predict_y)
    #print(predict_y.shape)
    for i in range(model_number):
        if predict_y[i][0] > predict_y[i][1]:
            predict_label[i] = 0
        else:
            predict_label[i] = 1

    cnt_label_1 = np.count_nonzero(predict_label == 0)
    cnt_label_2 = np.count_nonzero(predict_label == 1)

    if cnt_label_1 > cnt_label_2:
        determined_label = 0
    else:
        determined_label = 1

    if determined_label == data_y[label]:
        cnt += 1
    all_cnt += 1
    #if all_cnt == test_number:
    if i == test_number - 1:
        score = cnt / all_cnt
        print("score:",score)
        break
"""
    prediction_label = str(0)
#""""
    #予測ラベルを送信
    client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    print("send_label:",prediction_label)
    client.sendto(prediction_label.encode('utf-8'),(HOST,PORT))
    time.sleep(1.0)
#"""



#%%
print(cnt / all_cnt)
#print(len(y_test))
#print(y_test.shape)
#%%
