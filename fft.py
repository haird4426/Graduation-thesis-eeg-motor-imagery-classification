#%%
import numpy as np
from scipy.fftpack import fft,fftfreq,ifft
import matplotlib.pyplot as plt

#fs is sampling frequency
fs = 256.
time = np.linspace(0,3,int(10*fs),)#endpoint=False)
print(time)
#wave is the sum of sine wave(1Hz) and cosine wave(10 Hz)
wave = np.sin(2*5*np.pi*time)+np.cos(2*15*np.pi*time)

plt.plot(time,wave)
plt.xlabel("time (second)")
plt.show()

#%%
fft_wave = fft(wave)
fft_fre = fftfreq(n=wave.size, d=1/fs)

plt.plot(fft_fre,fft_wave.real,label="real part")
plt.show()

plt.plot(fft_fre,fft_wave.imag,label="imaginary part")
plt.xlabel("frequency (Hz)")
plt.show()

plt.plot(fft_fre,np.abs(fft_wave),label="abs")
plt.xlabel("frequency (Hz)")
plt.show()


#%%
#inverse fft
ifft_wave = ifft(fft_wave)

plt.subplot(311)
plt.plot(time,wave,alpha=0.3,linewidth=1,c="red",label="original wave")
plt.legend(loc=1)
plt.subplot(312)
plt.plot(time,ifft_wave.real,linewidth=1,label="real part")
plt.legend(loc=1)
plt.subplot(313)
plt.plot(time,ifft_wave.imag,label="imaginary part")
plt.legend(loc=1)
plt.show()




#%%
from scipy.fftpack import fft,fftfreq,ifft
import matplotlib.pyplot as plt
import numpy as np
#sampling frequency
fs = 256.
#time data, 10sec
time = np.linspace(0,10,int(10*fs))#endpoint=False)

wave = np.sin(2*np.pi*time)
#add noise
for i in range(150,300):
    wave += 0.05*np.random.rand()*np.sin(2*np.pi*i/10*(time+np.random.rand()))

plt.plot(time,wave)
plt.show()

fft_wave = fft(wave)
fft_fre = fftfreq(n=wave.size, d=1/fs)

plt.subplot(211)
plt.plot(fft_fre,fft_wave.real,label="real part")
plt.xlim(-50,50)
plt.ylim(-600,600)
plt.legend(loc=1)
plt.subplot(212)
plt.plot(fft_fre,fft_wave.imag,label="imaginary part")
plt.legend(loc=1)
plt.xlim(-50,50)
plt.ylim(-600,600)
plt.xlabel("frequency (Hz)")
plt.show()

#%%

#50Hz noach filter
fft_pass = np.where(np.abs(fft_fre)!=50,fft_fre,0)
fft_pass = np.where(fft_pass==0,fft_pass,1)
fft_pass[0] = 1

"""
#0.5Hzハイパスフィルター
fft_pass = np.where(np.abs(fft_fre)>0.5,fft_fre,0)
fft_pass = np.where(fft_pass==0,fft_pass,1)
"""
"""
#2~60Hz band pass filter
fft_pass = np.where((2<=np.abs(fft_fre)) & (np.abs(fft_fre)<=60),fft_fre,0)
fft_pass = np.where(fft_pass==0,fft_pass,1)
"""
plt.scatter(fft_fre,fft_pass,s=1,c="red")
plt.show()

#%%
fft_wave *= fft_pass

plt.subplot(211)
plt.plot(fft_fre,fft_wave.real,label="real part")
plt.xlim(-50,50)
plt.ylim(-600,600)
plt.legend(loc=1)
plt.subplot(212)
plt.plot(fft_fre,fft_wave.imag,label="imaginary part")
plt.legend(loc=1)
plt.xlim(-50,50)
plt.ylim(-600,600)
plt.xlabel("frequency (Hz)")
plt.show()

#%%
#inverse fft
ifft_wave = ifft(fft_wave)

plt.plot(time,wave,alpha=0.7,linewidth=1,c="red",label="original wave")
plt.plot(time,ifft_wave.real,linewidth=1,label="real part")
plt.xlim(0,2)
plt.ylim(-2,2)
plt.legend(loc=1)
plt.show()

#%%
