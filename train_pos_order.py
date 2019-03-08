import numpy as np
import codecs
import sys
import chainer
from model import Generate_RNN as Parse_Generate_RNN
from model import RNNUpdater
from chainer import training, datasets, iterators, optimizers
from chainer.training import extensions

batch_size = 10
uses_device = -1

cp = np
if uses_device >= 0:
	import cupy as cp

s = codecs.open("all-sentence-parses.txt", "r", "utf-8")
sentence = []

line = s.readline()
while line:
	one = [0]
	one.extend(list(map(int, line.split(","))))
	one.append(1)
	sentence.append(one)
	line = s.readline()
s.close()

n_words = max([max(l) for l in sentence]) + 1
l_max = max([len(l) for l in sentence])

for i in range(len(sentence)):
	sentence[i].extend([1] * (l_max - len(sentence[i])))

model = Parse_Generate_RNN(n_words, 20)

if uses_device >= 0:
	chainer.cuda.get_device_from_id(0).use()
	chianer.cuda.check_cuda_available()
	model.to_gpu()

optimizer = optimizers.Adam()
optimizer.setup(model)

train_iter = iterators.SerialIterator(sentence, batch_size, shuffle=False)

updater = RNNUpdater(train_iter, optimizer, device=uses_device, cp=cp)
trainer = training.Trainer(updater, (200, "epoch"))

trainer.extend(extensions.ProgressBar(update_interval=1))

trainer.run()

chainer.serializers.save_hdf5("pos_order.hdf5", model)




