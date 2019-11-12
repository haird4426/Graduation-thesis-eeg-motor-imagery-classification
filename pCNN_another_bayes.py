#%%
#pCNN
#データ増強を先にやっている
#データ読み込みを改良➡高速化
#ベイズ最適化を実装

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

stride = 0.1             #ストライド(s)
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
import random
random_label = []
for i in range(4):
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
filter = signal.firwin(numtaps=tap_number,cutoff=[2,60],fs=sampling_freq)
data_x[0:3,:all_sample_number] = signal.lfilter(filter,1,data_x[0:3,:all_sample_number])
#μ+-6σの修正外れ値にクリップ

#(ch_number,all_sample_number,extraction_freq)

#各チャンネルで(xi-μi)/σiで標準化
print("Normalized by (xi-μ) / σi for each channel")
avg = np.zeros((ch_number),dtype="float64")
std = np.zeros((ch_number),dtype="float64")
print("data_x.shape",data_x.shape)
data_x = data_x.reshape((ch_number,extraction_freq,all_sample_number))
#"""
for ch in range(ch_number):
    #for i in range(all_sample_number):
    #for i in range(extraction_freq):
        #avg[ch][i] = np.average(data_x[ch][i])
        #std[ch][i] = np.std(data_x[ch][i])
    avg[ch] = np.average(data_x[ch])
    std[ch] = np.std(data_x[ch])

for ch in range(ch_number):
    #for i in range(all_sample_number):
    #for i in range(extraction_freq):
        data_x[ch] = (data_x[ch] - avg[ch]) / std[ch]
#"""
#avg_1 = np.average(data_x)
#std_1 = np.std(data_x)
#data_x = (data_x - avg_1) / std_1
#ノイズ除去のためにreshape
data_x = data_x.reshape((all_sample_number,ch_number,extraction_freq))

#EOG,EMGアーティファクト除去
print("EOG,EMG artifact removal")
print("asn:",all_sample_number)
tmp = np.zeros(ch_number,dtype="float64")

##max<(x_ne) - min(x_ne)>_N
print("< max(x_ne) - min(x_ne) >_N")
delete_list = []
for asn in range(all_sample_number):
    tmp[0] = np.amax(data_x[asn][0]) - np.min(data_x[asn][0])
    tmp[1] = np.amax(data_x[asn][1]) - np.min(data_x[asn][1])
    tmp[2] = np.amax(data_x[asn][2]) - np.min(data_x[asn][2])
    #print(tmp)
    z = np.average(tmp)
    if z > threshold:
        delete_list.append(asn)
        all_sample_number -= 1
        #print(delete_list)

data_x = np.delete(data_x,delete_list,0)
data_y = np.delete(data_y,delete_list,0)
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
#----------------------------ハイパーパラメータ
over_lap_number = 128
nperseg_number  = 143
#----------------------------
"""
#論文のpCNNの値
over_lap_number = 112
nperseg_number  = 128
"""
#---------------------------

#--------------------------------input_dataの型を計算で求める
input_data  = np.zeros((ch_number,all_sample_number,72,70),dtype="complex128")

f, t, input_data[0:3,:all_sample_number] = scipy.signal.stft(data_x[0:3,:all_sample_number],fs=sampling_freq,nperseg=nperseg_number,noverlap=over_lap_number)

print("len(t):",len(t))
print("len(f:)",len(f))

input_data = abs(input_data)

print("input_data.shape:",input_data.shape)

"""
#max-minを確認➡STFTではせいぜい2.0もない➡正規化する価値あるかも
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
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
#--------------------------------------------
#gpuの必要なメモリしか使わない文
import tensorflow as tf
from keras.backend import tensorflow_backend
"""
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)
"""
#--------------------------------------------

from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Activation
from keras.optimizers import Adam,RMSprop,Adagrad,Adadelta,Adamax,Nadam
from keras.layers import BatchNormalization,Dropout
from keras.initializers import RandomNormal
from keras import regularizers

#optimizer = Adam(lr=10e-6)
#➡ベイズ最適化するときは関数内で宣言する
epoch = 3
dropout_rate = 0
strides = None #default None
pool_size = (2,2)
regularization_parameter = 0
batch_size = 256
lr = 1e-06

first_filter_number  = 100  #24
second_filter_number = 200  #48
third_filter_number  = 128  #96
fourth_filter_number = 256


first_kernel_size  = 12 #12
second_kernel_size = 13  #8
third_kernel_size  = 14  #4
fourth_kernel_size = 2

activation_function = "relu"
kf = KFold(n_splits=5, shuffle=True)
all_loss=[]
all_val_loss=[]
all_acc=[]
all_val_acc=[]
layer_number = 3
filter_number = [] * 4
kernel_size   = [] * 4
def cnn(#lr,#layer_number,
        #first_kernel_size,
        #second_kernel_size,
        #third_kernel_size,
        #fourth_kernel_size,
        #first_filter_number,
        #second_filter_number,
        third_filter_number,
        fourth_filter_number):
#def cnn(lr,layer_number,kernel_size,
#        filter_number):
    model = Sequential()
    #第１層目
    model.add(Conv2D(filters=int(first_filter_number),
                kernel_size=int(first_kernel_size),
                padding="same",
                bias_initializer="he_normal",
                kernel_initializer="he_normal",
                #bias_regularizer = regularizers.l2(regularization_parameter),
                kernel_regularizer = regularizers.l2(regularization_parameter),
                #activity_regularizer = regularizers.l2(regularization_parameter),
                input_shape=[len(f),len(t),ch_number]
    ))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=pool_size,strides=strides))
    model.add(Activation(activation_function))
    #第n層目
    for layer in range(int(layer_number)):
        if layer == 0:
            filter_number = second_filter_number
            kernel_size   = second_kernel_size
        elif layer == 1:
            filter_number = third_filter_number
            kernel_size   = third_kernel_size
        elif layer == 2:
            filter_number = fourth_filter_number
            kernel_size   = fourth_kernel_size
        #print(layer)
        model.add(Conv2D(filters=int(filter_number),
                kernel_size=int(kernel_size),
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

    model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(units=classes,activation="softmax"))

    model.compile(loss = "binary_crossentropy",
                    optimizer = Adam(lr),
                    metrics = ["accuracy"])
    #--------------------------------------------
    #gpuの必要なメモリしか使わない文
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    session = tf.Session(config=config)
    tensorflow_backend.set_session(session)
    #--------------------------------------------
    history = model.fit(x_train,     #train_data
                        y_train, #train_label
                        epochs=epoch,
                        batch_size=int(batch_size),
                        #validation_data=(x_test,y_test)
                        )
    """
    loss=history.history['loss']
    val_loss=history.history['val_loss']
    acc=history.history['acc']
    val_acc=history.history['val_acc']

    all_loss.extend(loss)
    all_val_loss.extend(val_loss)
    all_acc.extend(acc)
    all_val_acc.extend(val_acc)
    """
    score = model.evaluate(x_test,y_test,verbose=1)
    #return history.history['val_loss'] * (-1)
    print(score)
    #return score[0] * (-1)
    return score[1]

#model.summary()



#%%
#ベイズ最適化
from bayes_opt import BayesianOptimization
def bayesOpt():
    pbounds = {
        #"filter_number[0]" : (24,400),
        #"filter_number[1]" : (48,200),
        #"filter_number[2]" : (96,300),
        #"filter_number[3]" : (192,400),
        #"first_filter_number" : (100,150),
        #"first_kernel_size" : (10,15),
        "fourth_filter_number" : (100,400),
        #"fourth_kernel_size" : (2,15),
        #"lr" : (1e-7,1e-5),
        #"second_filter_number" : (100,200),
        #"second_kernel_size" : (2,15),
        "third_filter_number" : (100,300),
        #"third_kernel_size" : (2,15),
    }
    optimizer = BayesianOptimization(f=cnn, pbounds=pbounds)
    optimizer.maximize(init_points=5, n_iter=15, acq='ucb')
    return optimizer

result = bayesOpt()
print(result.max)



#%%
#----学習
kf = KFold(n_splits=5, shuffle=True)
all_loss=[]
all_val_loss=[]
all_acc=[]
all_val_acc=[]
for train_index, val_index in kf.split(x_train,y_train):

    train_data  = x_train[train_index]
    train_label = y_train[train_index]
    val_data    = x_train[val_index]
    val_label   = y_train[val_index]

    history = model.fit(train_data,
                        train_label,
                        epochs=epoch,
                        batch_size=batch_size,
                        validation_data=(val_data,val_label)
                        )

    loss=history.history['loss']
    val_loss=history.history['val_loss']
    acc=history.history['acc']
    val_acc=history.history['val_acc']

    all_loss.extend(loss)
    all_val_loss.extend(val_loss)
    all_acc.extend(acc)
    all_val_acc.extend(val_acc)



#%%
#----グラフ描画
#loss
#plt.figure(facecolor="azure", edgecolor="coral", linewidth=2)
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.plot(all_loss)
plt.plot(all_val_loss)

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#Accuracy
#plt.figure(facecolor="azure", edgecolor="coral", linewidth=2)
#plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
plt.plot(all_acc)
plt.plot(all_val_acc)

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#%%
