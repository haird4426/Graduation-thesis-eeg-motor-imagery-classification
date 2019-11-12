#%%
import numpy as np
from scipy.signal import windows
from matplotlib import pyplot as plt

#窓関数処理
N = 100
#ハニング窓
hanning_window = windows.hann(N)
#ハミング窓
hamming_window = windows.hamming(N)
#ブラックマン窓
black_window   = windows.blackmanharris(N)

#set numpy array
a = np.ones((2,2,N))
#print(a)

a = a * hanning_window
#print(a)

x = np.linspace(0,2*np.pi,N)
y = np.sin(x)
a[0][0] *= y

print(a)
plt.plot(x,y)
#plt.show()

y *= hanning_window
a *= hanning_window

plt.plot(x,a[0][0])
plt.show()
"""
plt.plot(x,hanning_window)
plt.plot(x,hamming_window)
plt.plot(x,black_window)
plt.show()
"""

