#%%
#----データセットの整形
import scipy.io
import numpy as np
from matplotlib import pyplot
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

stride = 0.125       #ストライド(s)
time_window = 4     #時間窓
threshold = 3       #ノイズ除去の閾値
#----------------------------------------------------------
each_data_number = 72 #各runファイルの試行回数
sampling_freq = 256   #サンプリング周波数
ch_number = 3         #チャンネル数
file_number = 3       #各被験者ディレクトリのファイル数
imagination_time = 4  #運動想起時間
label_number = 3      #ラベル数
delete_label = 3 - 1      #削除するラベル-1
start_time = 1.5    #抽出する時間
end_time = 8.5      #抽出する時間

data_magnification = int(((end_time - time_window - start_time) / stride) + 1) #データ増加率
print("data_mag:",data_magnification)

imagination_freq = sampling_freq * imagination_time #運動想起時間→周波数変換time_windowの方がよいか
#stride_freq = int(sampling_freq * stride) #ストライド時間→周波数変換
all_sample_number = int(each_data_number * file_number * subject_number * data_magnification * 2 / 3)
print("asn:",all_sample_number)
#-------------------------------------------------
#print(os.getcwd()) #カレントディレクトリ
#相対パスで全ファイルにアクセス
current_path = os.getcwd() #これでPCによらない
#print("current_path:",current_path)
#-------------------------------------------------

data_ch1 = np.zeros((all_sample_number,imagination_freq),dtype="float64") #ch1,run1
data_ch2 = np.zeros((all_sample_number,imagination_freq),dtype="float64") #ch2,run1
data_ch3 = np.zeros((all_sample_number,imagination_freq),dtype="float64") #ch3,run1

data_label = np.zeros(all_sample_number)

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

    starting_imagination_label = np.zeros(file_number,dtype="int64")
    starting_imagination_time  = np.zeros(file_number,dtype="float32")
    label = 0
    for j in range(each_data_number): #ラベル数のループ回数→すべてのtts分を見る
        label_jdg = np.zeros((file_number,label_number),dtype="int8")
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
        if j % 10 == 0:
            print("j:",j)
        for k in a:
            #stride_label = 0
            k = int(k * sampling_freq)
            #print("k",k)

            if label_jdg[0][delete_label] != 1:
                data_label[run_cnt+label] = data_run1["Y"][0][j]
                for l in range(imagination_freq):
                    data_ch1[run_cnt+label][l] = data_run1["X"][starting_imagination_label[0]-k+l][0]
                    data_ch2[run_cnt+label][l] = data_run1["X"][starting_imagination_label[0]-k+l][1]
                    data_ch3[run_cnt+label][l] = data_run1["X"][starting_imagination_label[0]-k+l][2]
                #stride_label += 1
                label += 1

            if label_jdg[1][delete_label] != 1:
                #stride_label = 0
                data_label[run_cnt+label] = data_run2["Y"][0][j]
                for l in range(imagination_freq):
                    data_ch1[run_cnt+label][l] = data_run2["X"][starting_imagination_label[1]-k+l][0]
                    data_ch2[run_cnt+label][l] = data_run2["X"][starting_imagination_label[1]-k+l][1]
                    data_ch3[run_cnt+label][l] = data_run2["X"][starting_imagination_label[1]-k+l][2]
                #stride_label += 1
                label += 1

            if label_jdg[2][delete_label] != 1:
                #stride_label = 0
                data_label[run_cnt+label] = data_run3["Y"][0][j]
                for l in range(imagination_freq):
                    data_ch1[run_cnt+label][l] = data_run3["X"][starting_imagination_label[2]-k+l][0]
                    data_ch2[run_cnt+label][l] = data_run3["X"][starting_imagination_label[2]-k+l][1]
                    data_ch3[run_cnt+label][l] = data_run3["X"][starting_imagination_label[2]-k+l][2]
                #stride_label += 1
                label += 1
    run_cnt += int(each_data_number * 3 * 2 / 3)

#"""
#グラフ表示
x = np.linspace(0,imagination_time,imagination_freq)
import random
random_label = []
for i in range(5):
    random_label.append(random.randint(0,all_sample_number))
    y = data_ch1[random_label[i]] #ランダムに表示
    pyplot.plot(x,y)
    print("label",random_label[i])
    pyplot.show()
#"""
#"""
#0の要素があるかチェック➡OK:ファイルの名前を変更して対処
print("-----")
for label in range(all_sample_number):
    for j in range(imagination_freq):
        if data_ch1[label][j] == 0:
            print("input_label:",label)
            break
    if data_label[label] == 0:
        print("output_label:",label)
#"""



#%%
#----脳波の前処理
import scipy.signal as signal

filter = [0] * ch_number
tap_number = 61

#"""
#50Hzノッチフィルター
print("50Hz notch filter")
for i in range(all_sample_number):
    filter[0] = signal.firwin(numtaps=tap_number,cutoff=[49,51],fs=sampling_freq)
    filter[1] = signal.firwin(numtaps=tap_number,cutoff=[49,51],fs=sampling_freq)
    filter[2] = signal.firwin(numtaps=tap_number,cutoff=[49,51],fs=sampling_freq)

    data_ch1[i] = signal.lfilter(filter[0],1,data_ch1[i])
    data_ch2[i] = signal.lfilter(filter[1],1,data_ch2[i])
    data_ch3[i] = signal.lfilter(filter[2],1,data_ch3[i])
#"""

#"""
#カットオフ周波数0.5Hzのハイパスフィルター
print("High-pass filter with a cutoff frequency of 0.5Hz")
for i in range(all_sample_number):
    filter[0] = signal.firwin(numtaps=tap_number,cutoff=0.5,fs=sampling_freq)
    filter[1] = signal.firwin(numtaps=tap_number,cutoff=0.5,fs=sampling_freq)
    filter[2] = signal.firwin(numtaps=tap_number,cutoff=0.5,fs=sampling_freq)

    data_ch1[i] = signal.lfilter(filter[0],1,data_ch1[i])
    data_ch2[i] = signal.lfilter(filter[1],1,data_ch2[i])
    data_ch3[i] = signal.lfilter(filter[2],1,data_ch3[i])
#"""

#"""
#5次の0位相バターワースフィルターで2~60Hzのバンドパスフィルター
print("2-60Hz band pass filter")
#------------------------------------バターワースフィルターを要実装
for i in range(all_sample_number):
    filter[0] = signal.firwin(numtaps=tap_number,cutoff=[2,60],fs=sampling_freq)
    filter[1] = signal.firwin(numtaps=tap_number,cutoff=[2,60],fs=sampling_freq)
    filter[2] = signal.firwin(numtaps=tap_number,cutoff=[2,60],fs=sampling_freq)

    data_ch1[i] = signal.lfilter(filter[0],1,data_ch1[i])
    data_ch2[i] = signal.lfilter(filter[1],1,data_ch2[i])
    data_ch3[i] = signal.lfilter(filter[2],1,data_ch3[i])
#"""

#μ+-6σの修正外れ値にクリップ

#"""
#各チャンネルで(xi-μi)/σiで正規化
print("Normalized by (xi-μ) / σi for each channel")
avg_ch1 = np.average(data_ch1)
avg_ch2 = np.average(data_ch2)
avg_ch3 = np.average(data_ch3)

std_ch1 = np.std(data_ch1)
std_ch2 = np.std(data_ch2)
std_ch3 = np.std(data_ch3)

data_ch1 = (data_ch1 - avg_ch1) / std_ch1
data_ch2 = (data_ch2 - avg_ch2) / std_ch2
data_ch3 = (data_ch3 - avg_ch3) / std_ch3
#"""

#-----------------------------------------------------------------------------
#i番目のデータを削除する関数
def delete_data(i,all_sample_number,data_ch1,data_ch2,data_ch3,data_label):
    #print("old:",data_ch1.shape)
    data_ch1 = np.delete(data_ch1,i,0)
    data_ch2 = np.delete(data_ch2,i,0)
    data_ch3 = np.delete(data_ch3,i,0)

    data_label = np.delete(data_label,i,0)

    #print("new:",data_ch1.shape)
    #print("label:",data_label.shape)
    all_sample_number -= 1
    i -= 1
    #print("asn:",all_sample_number)
    #print("i_2:",i)
    return i,all_sample_number,data_ch1,data_ch2,data_ch3,data_label
#-----------------------------------------------------------------------------
#"""
#EOG,EMGアーティファクト除去
print("EOG,EMG artifact removal")
print("asn:",all_sample_number)
tmp = np.zeros(ch_number,dtype="float32")

##max<(x_ne) - min(x_ne)>_N
print("< max(x_ne) - min(x_ne) >_N")
i = 0
while i < all_sample_number:
#for i in range(all_sample_number):
    #i -= 1
    #print("i_4:",i)
    tmp[0] = np.amax(data_ch1[i]) - np.min(data_ch1[i])
    tmp[1] = np.amax(data_ch2[i]) - np.min(data_ch2[i])
    tmp[2] = np.amax(data_ch3[i]) - np.min(data_ch3[i])

    z = np.average(tmp)
    #print("tmp:",tmp)
    #if tmp[0] > threshold or tmp[1] > threshold or tmp[2] > threshold:
    if z > 3:
        i,all_sample_number,data_ch1,data_ch2,data_ch3,data_label = delete_data(i,all_sample_number,data_ch1,data_ch2,data_ch3,data_label)
        #print("i_2:",i)
    if i + 1 == all_sample_number or i + 2 == all_sample_number:
        break
    #print("i_3:",i)
    i = i + 1
print("asn:",all_sample_number)

##<x_ne> - <x_n>
print("<x_ne> - <x_n>")
avg_ch_epoch = np.zeros(ch_number,dtype="float32")
avg_ch       = np.zeros(ch_number,dtype="float32")
while i < all_sample_number:
#for i in range(all_sample_number):
    avg_ch_epoch[0] = np.average(data_ch1[i])
    avg_ch_epoch[1] = np.average(data_ch2[i])
    avg_ch_epoch[2] = np.average(data_ch3[i])

    avg_ch[0] = np.average(data_ch1)
    avg_ch[1] = np.average(data_ch2)
    avg_ch[2] = np.average(data_ch3)

    tmp[0] = np.abs(avg_ch_epoch[0] - avg_ch[0])
    tmp[1] = np.abs(avg_ch_epoch[1] - avg_ch[1])
    tmp[2] = np.abs(avg_ch_epoch[2] - avg_ch[2])

    if tmp[0] > threshold or tmp[1] > threshold or tmp[2] > threshold:
        i,all_sample_number,data_ch1,data_ch2,data_ch3,data_label = delete_data(i,all_sample_number,data_ch1,data_ch2,data_ch3,data_label)

    if i + 1 == all_sample_number:
        break
    i += 1
print("asn:",all_sample_number)

##S^2_x_ne
print("S^2_x_ne")
while i < all_sample_number:
#for i in range(all_sample_number):
    tmp[0] = np.var(data_ch1[i])
    tmp[1] = np.var(data_ch2[i])
    tmp[2] = np.var(data_ch3[i])

    if tmp[0] > threshold or tmp[1] > threshold or tmp[2] > threshold:
        i,all_sample_number,data_ch1,data_ch2,data_ch3,data_label = delete_data(i,all_sample_number,data_ch1,data_ch2,data_ch3,data_label)

    if i + 1 == all_sample_number:
        break
    i += 1
print("asn:",all_sample_number)

##median(d(x_ne) / dt)
print("median(d(x_ne) / dt)")
dt = float(1 / sampling_freq)
while i < all_sample_number:
#for i in range(all_sample_number):
    tmp[0] = np.median(np.diff(data_ch1[i]) / dt)
    tmp[1] = np.median(np.diff(data_ch2[i]) / dt)
    tmp[2] = np.median(np.diff(data_ch3[i]) / dt)
    if tmp[0] > threshold or tmp[1] > threshold or tmp[2] > threshold or tmp[0] < -threshold or tmp[1] < -threshold or tmp[2] < -threshold:
        i,all_sample_number,data_ch1,data_ch2,data_ch3,data_label = delete_data(i,all_sample_number,data_ch1,data_ch2,data_ch3,data_label)

    if i + 1 == all_sample_number:
        break
    i += 1
print("asn:",all_sample_number)


#グラフ描画
#-----------------1セル目と同様にするようにしたほうがよいか
random_label = []
for i in range(5):
    random_label.append(random.randint(0,all_sample_number))
    y = data_ch1[random_label[i]] #ランダムに表示
    pyplot.plot(x,y)
    pyplot.show()



#%%
#----データの整理
from sklearn import model_selection
from keras.utils import np_utils

#input_data = np.zeros((ch_number,all_sample_number,imagination_freq),dtype="float64")
input_data = []
input_data.append(data_ch1)
input_data.append(data_ch2)
input_data.append(data_ch3)
input_data = np.array(input_data)
print("input_data.shape:",input_data.shape)

input_data = input_data.reshape(all_sample_number,ch_number,imagination_freq,1)
classes = 2
test_rate = 0.2

#正解ラベルYを(each_data_number,)へ
print(data_label.shape)
print("input_data.shape:",input_data.shape)

#出力Yのラベルを1,2,3から0,1,2に変更
for i in range(len(data_label)):
    if data_label[i] == 1:
        data_label[i] = 0
    elif data_label[i] == 2:
        data_label[i] = 1
    elif data_label[i] == 3:
        data_label[i] = 2
#print(data_label) #OK

#正解ラベルYをont-hot表現へ
data_label = np_utils.to_categorical(data_label,classes)
#print(data_label) #OK

#以下にモデルの検証の仕方
#https://newtechnologylifestyle.net/%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%80%81%E3%83%87%E3%82%A3%E3%83%BC%E3%83%97%E3%83%A9%E3%83%BC%E3%83%8B%E3%83%B3%E3%82%B0%E3%81%A7%E3%81%AE%E5%AD%A6%E7%BF%92%E3%83%87%E3%83%BC%E3%82%BF%E3%81%A8/

####データの分割####
x_train, x_test, y_train, y_test = model_selection.train_test_split(input_data, data_label, test_size=test_rate)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)



#%%
#dCNNモデル構築
from keras.models import Sequential
from keras.layers import Reshape,Dense,Conv2D,Conv3D,MaxPooling2D,MaxPooling3D,Flatten,Activation
from keras.optimizers import Adam
from keras.layers import BatchNormalization,Dropout
from keras.initializers import RandomNormal

dropout_rate = 0.1
model = Sequential()
first_filter_number  = 25
second_filter_number = 25
third_filter_number  = 50
fourth_filter_number = 100
fifth_filter_number  = 200

first_kernel_size  = (1,10)
second_kernel_size = (3,1,25)
third_kernel_size  = (10,25)
fourth_kernel_size = (10,50)
fifth_kernel_size  = (10,100)

activation_function = "relu"
poolsize = (3,1)

model = Sequential()

#1層
model.add(Conv2D(filters=first_filter_number,
            kernel_size=first_kernel_size,
            #padding="same",                #sameじゃないと2層の形状がわからない
            bias_initializer="he_normal",
            kernel_initializer="he_normal",
            input_shape=(ch_number,imagination_freq,1)))
model.add(BatchNormalization())
model.add(Activation(activation_function))

#2層
#print(model.output_shape)

model.add(Reshape((3,model.output_shape[2],first_filter_number,1)))#,input_shape=(ch_number,imagination_freq,first_filter_number)))

#print(model.output_shape)

model.add(Conv3D(filters=second_filter_number,
            kernel_size=second_kernel_size,
            #padding="same",
            bias_initializer="he_normal",
            kernel_initializer="he_normal"))

#print(model.output_shape)

model.add(BatchNormalization())
model.add(Activation(activation_function))
model.add(MaxPooling3D(pool_size=(1,3,1)))
#model.add(Dropout(dropout_rate))

#print(model.output_shape)

#3層
a = model.output_shape[3] * model.output_shape[4]
model.add(Reshape((model.output_shape[2],a,model.output_shape[1])))

#print(model.output_shape)

model.add(Conv2D(filters=third_filter_number,
            kernel_size=third_kernel_size,
            #padding="same",
            bias_initializer="he_normal",
            kernel_initializer="he_normal"))
model.add(BatchNormalization())
model.add(Activation(activation_function))
model.add(MaxPooling2D(pool_size=poolsize))
#model.add(Dropout(dropout_rate))

print(model.output_shape)

#"""
#4層
model.add(Reshape((model.output_shape[1],model.output_shape[3],model.output_shape[2])))
print(model.output_shape)

model.add(Conv2D(filters=fourth_filter_number,
            kernel_size=fourth_kernel_size,
            #padding="same",
            bias_initializer="he_normal",
            kernel_initializer="he_normal"))
model.add(BatchNormalization())
model.add(Activation(activation_function))
model.add(MaxPooling2D(pool_size=poolsize))
#model.add(Dropout(dropout_rate))
#"""
print(model.output_shape)
#"""
#5層
model.add(Reshape((model.output_shape[1],model.output_shape[3],model.output_shape[2])))
print(model.output_shape)

model.add(Conv2D(filters=fifth_filter_number,
            kernel_size=fifth_kernel_size,
            #padding="same",
            bias_initializer="he_normal",
            kernel_initializer="he_normal"))
model.add(BatchNormalization())
model.add(Activation(activation_function))
model.add(MaxPooling2D(pool_size=poolsize))
#model.add(Dropout(dropout_rate))
#"""
print(model.output_shape)

model.add(Flatten())
model.add(Dense(units=classes,activation="softmax"))

model.compile(loss = "binary_crossentropy",
                optimizer = Adam(),
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
                    epochs=50,
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
