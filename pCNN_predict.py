#%%
import numpy as np
from keras.models import model_from_json

#モデル保存
open("cnn.json","w").write(model.to_json())

#学習済みの重みを保存
model.save_weights("cnn_weight.h5")

#モデル読み込み
model = model_from_json(open("cnn.json","r").read())

#重み読み込み
model.load_weights("cnn_weight.hdf5")

#読み込んだ学習済みモデルで予測
y = model.predict(np.)