#Brain Data Preprocessing module

"""
窓関数処理
fft
min~max band pass filter
50Hz noach filter
(0.5Hz high pass filter)
ifft
μ+-6σの修正外れ値にクリップ(未実装)←いらないかも
Normalized by (x_i - μ) / σ_i for each channnel(line,col,all,each channnel)
denoising
"""

