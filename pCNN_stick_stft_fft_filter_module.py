#%%
#pCNN

#stftされた結果を結合する実験

#データ増強を先にやっている
#データ読み込みを改良➡高速化
#unityへデータを送るのも実装
#前処理を自作で実装
#交差検証なし➡readsumよりも低く過学習する
#標準化方向実験➡あまり変わらない
#窓関数処理を追加
#フィルター処理，stftでrealに変更 (absでも精度はさほど変化しないがifftで復元されない)
#モジュール構造に変更
#固定パラメータをモジュール間で共有化

import numpy as np
from matplotlib import pyplot as plt
from Read_Brain_Data import read_brain_data
from Random_Gragh import random_gragh
import param

#被験者を指定
print("subject"+str(param.subject))

print("data_magnigication:",param.data_magnification)

#脳波データの型を初期化
data_x = np.zeros((param.ch_number,param.all_sample_number,param.extraction_freq),dtype="float64")
data_y = np.zeros(param.all_sample_number,dtype="int32")
print("data_x.shape",data_x.shape)

run_cnt = 0
for i in range(param.subject_number): #各被験者ごとのループ
    read_brain_data(data_x,data_y,run_cnt)

#グラフ描画
x = np.linspace(0,param.imagination_time,param.extraction_freq)
random_gragh(x,data_x)



#%%
#----前処理
from scipy.fftpack import fft,fftfreq,ifft

fft_wave_complex = np.zeros((param.ch_number,param.all_sample_number,param.extraction_freq),dtype="complex128")

"""
f = fft(data_x[0][0])
g = fftfreq(n=data_x[0][0].size, d=1/sampling_freq)
print(f.shape)
print(g.shape)
"""

a = 1
plt.plot(x,data_x[0][a])
plt.title("label %d"%a)
plt.show()

#window function process
window_function = param.hamming_window
data_x *= window_function
#data_x[:][:] /= np.amax(data_x[0][a]) #標準化して窓関数と一致するか検証


plt.plot(x,data_x[0][a])
plt.plot(x,window_function)
plt.show()


#fft
fft_wave_complex = fft(data_x)
fft_fre = fftfreq(n=data_x[0][0].size, d=1/param.sampling_freq)

"""
ifft ➡ 復元されるかの実験 ➡ realでやらなければ復元されない
fft_wave_abs = abs(fft_wave_complex)   #absはまちがいifftで復元されない
data_x[:][:] = ifft(fft_wave_abs[:][:])
data_x[:][:] = np.real(ifft(fft_wave_complex))
plt.plot(x,data_x[0][0])
plt.show()
"""

#周波数で正の値をとるインデックスをとる➡フーリエ変換した結果をみるため
pidxs = np.where(fft_fre >= 0)
#print(pidxs)

freq, power = fft_fre[pidxs], np.abs(fft_wave_complex[0][a])[pidxs]
plt.plot(freq,power)
plt.title("no filter")
plt.show()

#2~60Hzバンドパスフィルター
#-------------------------------------
max_freq = 30
min_freq = 2
#-------------------------------------
#print("%d-%dHz band pass filter"% (min_freq,max_freq))
fft_pass = np.where((min_freq<=np.abs(fft_fre)) & (np.abs(fft_fre)<=max_freq),fft_fre,0)
fft_pass = np.where(fft_pass==0,fft_pass,1)
fft_wave_complex[:][:] *= fft_pass

freq, power = fft_fre[pidxs], np.abs(fft_wave_complex[0][a])[pidxs]
plt.plot(freq,power)
plt.title("%d-%dHz band pass filter"% (min_freq,max_freq))
plt.show()

#50Hz noach filter ➡50Hz付近の値を除去できていない（電源ではないのか）
#print("50Hz notch filter")
fft_pass = np.where(np.abs(fft_fre)!=50,fft_fre,0)

fft_pass = np.where(fft_pass==0,fft_pass,1)
fft_pass[0] = 1
fft_wave_complex[:][:] *= fft_pass

freq, power = fft_fre[pidxs], np.abs(fft_wave_complex[0][a])[pidxs]
plt.plot(freq,power)
plt.title("50Hz noach filter")
plt.show()

"""
#0.5Hzハイパスフィルター
print("High-pass filter with a cutoff frequency of 0.5Hz")
fft_pass = np.where(np.abs(fft_fre)>0.5,fft_fre,0)
fft_pass = np.where(fft_pass==0,fft_pass,1)
fft_wave_complex[:][:] *= fft_pass
"""

#ifft
fft_wave_abs = abs(fft_wave_complex)
#data_x[:][:] = ifft(fft_wave_abs[:][:])
data_x[:][:] = np.real(ifft(fft_wave_complex))

#μ+-6σの修正外れ値にクリップ

#各チャンネルで(xi-μi)/σiで標準化
print("Normalized by (xi-μ) / σi for each channel")
avg = np.zeros((param.ch_number),dtype="float64")
std = np.zeros((param.ch_number),dtype="float64")
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
#    data_x[ch] = (data_x[ch] - avg[ch]) / std[ch]
#print(np.average(data_x[0]))
#print(np.std(data_x[0]))

#"""
#列
#data_x = data_x.reshape((ch_number,all_sample_number,extraction_freq))

#全体
data_x = ( data_x[:] - np.average(data_x[:]) ) / np.std(data_x[:])
#print(np.average(data_x))
#print(np.std(data_x))

"""
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
"""

#グラフ描画
y = data_x[0][a] #ランダムに表示
plt.plot(x,y)
plt.title("label %d"%a)
plt.show()
random_gragh(x,data_x)


#%%
#----STFT
from scipy.signal import stft

height = param.input_data_height
width  = param.input_data_width

#--------------------------------input_dataの型を計算で求める➡求める必要ないか？
input_data  = np.zeros((param.ch_number,param.all_sample_number,height,width),dtype="complex128")
input_data_stick = np.zeros((param.all_sample_number,height*3,width),dtype="float64")

f, t, input_data[0:3,:param.all_sample_number] = stft(data_x[0:3,:param.all_sample_number],fs=param.sampling_freq,nperseg=param.nperseg_number,noverlap=param.over_lap_number)

print("len(t):",len(t))
print("len(f:)",len(f))

f_stick = f
f_stick = np.append(f_stick,f+128)
f_stick = np.append(f_stick,f+256)

#print(f_stick)

#input_data = abs(input_data)
input_data = np.real(input_data) #➡realでないと復元されないはず，absでも精度は出る

input_data_stick[:,0:65,0:65]    = input_data[0,:,0:65,0:65]
input_data_stick[:,65:130,0:65]  = input_data[1,:,0:65,0:65]
input_data_stick[:,130:195,0:65] = input_data[2,:,0:65,0:65]

#stft描画
fig = 5
i = 0
j = 0
#plt.pcolormesh(t,f,input_data[i][j],vmin=0)
plt.pcolormesh(t,f_stick,input_data_stick[i],vmin=0)
plt.ylim([f[1],f[-1]])
plt.title("STFT Magnitude")
plt.legend(loc="upper right",fontsize=fig,title="input_data[%d][%d]"%(i,j))
plt.xlabel("Time[sec]")
plt.ylabel("Frequency[Hz]")
pp=plt.colorbar (orientation="vertical") # カラーバーの表示
pp.set_label("Label", fontname="Arial", fontsize=24) #カラーバーのラベル
plt.show()

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

input_data = input_data.reshape(param.all_sample_number,len(f),len(t),param.ch_number)
input_data_stick = input_data_stick.reshape((param.all_sample_number,len(f)*param.ch_number,len(t),1))
print("input_data.shape:",input_data.shape)
print("input_data_stick.shape:",input_data_stick.shape)



#%%
#----データ整理
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical

#正解ラベルYを(each_data_number,)へ
print(data_y.shape)
print(input_data_stick.shape)

#出力Yのラベルを1,2,3から0,1,2に変更
for i in range(len(data_y)):
    if data_y[i] == 1:
        data_y[i] = 0
    elif data_y[i] == 2:
        data_y[i] = 1
    elif data_y[i] == 3:
        data_y[i] = 2

#正解ラベルYをont-hot表現へ
data_y = to_categorical(data_y,param.classes)

#データの分割
x_train, x_test, y_train, y_test = train_test_split(input_data_stick, data_y, test_size=param.test_rate)

print("x_train:",x_train.shape)
print("y_train:",y_train.shape)
print("x_test:",x_test.shape)
print("y_test:",y_test.shape)



#%%
#----pCNNモデル構築
from keras.optimizers import Adam,RMSprop,Adagrad,Adadelta,Adamax,Nadam
from CNN import create_cnn_model

optimizer = Adam(lr=param.lr)

model = create_cnn_model(optimizer,input_data_stick[0].shape)
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
                    epochs=param.epoch,
                    batch_size=256,
                    validation_data=(x_test,y_test),
                    )

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
                        epochs=param.epoch,
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
#----未知データで検証
from keras.models import model_from_json

test_number = len(y_test)
model_number = 1
file_number = str(1)
predict_y = np.zeros((model_number,param.classes))
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
    predict_y[0] = model.predict(x_test[label].reshape(1,len(f),len(t),param.ch_number))
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
predict_y = np.zeros((model_number,param.classes))
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
    predict_y[0] = model_6.predict(x_test[label].reshape(1,len(f),len(t),param.ch_number))
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
