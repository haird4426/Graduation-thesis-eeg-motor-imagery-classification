import random
from matplotlib import pyplot as plt
import numpy as np
import param

gragh_number = 5
def random_gragh(x,data_x):
    random_label = []
    for i in range(gragh_number):
        random_label.append(random.randint(0,param.all_sample_number-1))
        y = data_x[0][random_label[i]] #ランダムに表示
        plt.plot(x,y)
        plt.title("label %d"%random_label[i])
        plt.show()