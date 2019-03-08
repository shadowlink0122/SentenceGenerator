import numpy as np
import codecs
import chainer
from model import Generate_RNN, RNNUpdater
from chainer import training, datasets, iterators, optimizers
from chainer.training import extensions

batch_size = 10
uses_device = -1

cp = np
if uses_device >= 0:
	import cupy as cp

# ファイルを読み込む
s = codecs.open("all-sentence.txt", "r", "utf-8")
sentence = []

# 行の中の単語を数字のリストにする。
# 行が終わると終端文字を入れ、新しい文を追加。
line = s.readline()
while line:
	one = [0]
	one.extend(list(map(int, line.split(","))))
	one.append(1)
	sentence.append(one)
	line = s.readline()
s.close()

# 単語の種類数
word_size = max([max(l) for l in sentence]) + 1
# 最長の文の長さ
l_max = max([len(l) for l in sentence])

# バッチ処理で、全ての文の長さを揃える。
for i in range(len(sentence)):
	sentence[i].extend([1] * (l_max - len(sentence[i])))

# モデルの生成
model = Generate_RNN(word_size, 200)
if uses_device >= 0:
	chainer.cuda.get_device_form_id(0).use()
	chainer.cuda.check_cuda_available()
	model.to_gpu()

# 逆伝播の方法はAdam
optimizer = optimizers.Adam()
optimizer.setup(model)

train_iter = iterators.SerialIterator(sentence, batch_size, shuffle=False)

# デバイス(CPU or GPU)を選択し、トレーナーを作成
updater = RNNUpdater(train_iter, optimizer, device=uses_device, cp=cp)
trainer = training.Trainer(updater, (30, "epoch"))

# 進行状況を表示
trainer.extend(extensions.ProgressBar(update_interval=1))

# 学習開始
trainer.run()

# 学習モデルの保存
chainer.serializers.save_hdf5("lang_model.hdf5", model)







