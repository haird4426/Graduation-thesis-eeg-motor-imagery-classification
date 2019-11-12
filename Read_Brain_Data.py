import scipy.io
import numpy as np
from Get_Nearest_Value import get_nearest_value
import param

def read_brain_data(data_x,data_y,run_cnt):
    #毎回各被験者のデータを読み込む
    data_run1 = (scipy.io.loadmat(param.current_path + "/s" + str(param.subject) + "/" + "/Run1.mat"))
    data_run2 = (scipy.io.loadmat(param.current_path + "/s" + str(param.subject) + "/" + "/Run2.mat"))
    data_run3 = (scipy.io.loadmat(param.current_path + "/s" + str(param.subject) + "/" + "/Run3.mat"))

    #ttsとtsの型を変更
    data_run1["trial_time_stamps"] = data_run1["trial_time_stamps"].reshape(param.each_data_number)
    data_run2["trial_time_stamps"] = data_run2["trial_time_stamps"].reshape(param.each_data_number)
    data_run3["trial_time_stamps"] = data_run3["trial_time_stamps"].reshape(param.each_data_number)

    data_run1["time_stamps"] = data_run1["time_stamps"].reshape(data_run1["time_stamps"].size)
    data_run2["time_stamps"] = data_run2["time_stamps"].reshape(data_run2["time_stamps"].size)
    data_run3["time_stamps"] = data_run3["time_stamps"].reshape(data_run3["time_stamps"].size)

    starting_imagination_label = np.zeros(param.file_number,dtype="int32")
    starting_imagination_time  = np.zeros(param.file_number,dtype="float64")
    label = 0
    for j in range(param.each_data_number): #ラベル数のループ回数→すべてのtts分を見る
        label_jdg = np.zeros((param.file_number,param.label_number),dtype="int32")
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
        starting_imagination_time[0]  = get_nearest_value(data_run1["time_stamps"],data_run1["trial_time_stamps"][j])
        starting_imagination_time[1]  = get_nearest_value(data_run2["time_stamps"],data_run2["trial_time_stamps"][j])
        starting_imagination_time[2]  = get_nearest_value(data_run3["time_stamps"],data_run3["trial_time_stamps"][j])

        starting_imagination_label[0] = np.where(data_run1["time_stamps"]==get_nearest_value(data_run1["time_stamps"],data_run1["trial_time_stamps"][j]))[0]
        starting_imagination_label[1] = np.where(data_run2["time_stamps"]==get_nearest_value(data_run2["time_stamps"],data_run2["trial_time_stamps"][j]))[0]
        starting_imagination_label[2] = np.where(data_run3["time_stamps"]==get_nearest_value(data_run3["time_stamps"],data_run3["trial_time_stamps"][j]))[0]

        #データ増強しながら取り込む
        a = [param.stride * x for x in range(param.data_magnification-1,-1,-1)]
        for k in a:
            k = int(k * param.sampling_freq)
            if label_jdg[0][param.delete_label] != 1:
                for ch in range(param.ch_number):
                    head = starting_imagination_label[0]-k
                    tail = starting_imagination_label[0]-k+param.extraction_freq
                    data_x[ch][run_cnt+label] = data_run1["X"][head:tail,ch]
                data_y[run_cnt+label] = data_run1["Y"][0][j]
                label += 1
            if label_jdg[1][param.delete_label] != 1:
                for ch in range(param.ch_number):
                    head = starting_imagination_label[1]-k
                    tail = starting_imagination_label[1]-k+param.extraction_freq
                    data_x[ch][run_cnt+label] = data_run2["X"][head:tail,ch]
                data_y[run_cnt+label] = data_run2["Y"][0][j]
                label += 1
            if label_jdg[2][param.delete_label] != 1:
                for ch in range(param.ch_number):
                    head = starting_imagination_label[2]-k
                    tail = starting_imagination_label[2]-k+param.extraction_freq
                    data_x[ch][run_cnt+label] = data_run3["X"][head:tail,ch]
                data_y[run_cnt+label] = data_run3["Y"][0][j]
                label += 1
    run_cnt += int(param.each_data_number * 2 * param.data_magnification)