import numpy as np
import codecs
import sys
import chainer
from model import Generate_RNN, RNNUpdater
from chainer import training, datasets, iterators, optimizers
import chainer.functions as F
from chainer.training import extensions

uses_device = -1

cp = np
if uses_device >= 0:
	import cupy as cp
	import chainer.cuda

# sys.stdout = codecs.getwriter("utf-8")(sys.stdout)

w = codecs.open("all-words.txt", "r", "utf-8")
words = {}

# 行の中の単語をリスト化
line = w.readline()
while line:
	l = line.split(",")
	if len(l) >= 2:
		words[int(l[0])] = l[1].strip()
	line = w.readline()
w.close()

# モデルの生成
model = Generate_RNN(len(words)+2, 200)

# 学習結果の読み込み
chainer.serializers.load_hdf5("lang_model.hdf5", model)

if uses_device >= 0:
	chainer.cuda.get_device_from_id(0).use()
	chainer.cuda.check_cuda_available()
	model.to_gpu()

words_max = 50
beam_w = 10
sentence = []
model_history = [model]
cur_sentence = [0]
cur_score = []
max_score = 0

def Tree_Traverse():
	global max_score

	cur_word = cur_sentence[-1]
	score = np.prod(cur_score)
	deep = len(cur_sentence)

	# 枝狩り
	if deep > 5 and max_score*0.6 > score:
		return

	# 終端文字または最大の文の長さ以上なら、文を追加して終了
	if cur_word == 1 or deep > words_max:
		data = np.array(cur_sentence)
		sentence.append((score, data))
		if max_score < score:
			max_score = score
		return

	cur_model = model_history[-1].copy()
	x = cp.array([cur_word], dtype=cp.int32)
	y = cur_model(x)
	z = F.softmax(y)
	result = z.data[0]

	if uses_device >= 0: result = chainer.cuda.to_cpu(result)
	p = np.argsort(result)[::-1]

	model_history.append(cur_model)
	for i in range(beam_w):
		cur_sentence.append(p[i])
		cur_score.append(result[p[i]])
		Tree_Traverse()
		cur_sentence.pop()
		cur_score.pop()
	model_history.pop()

# 木探索で探索で文章を作成
Tree_Traverse()

# スコアの高いものから順に表示
result_set = sorted(sentence, key=lambda x: x[0])[::-1]

for i in range(min([20, len(result_set)])):
	s, l = result_set[i]
	r = str(s) + "\t"
	for w in l:
		if w > 1:
			r += words[w]
	r += "\n"

	print(r)

	# sys.stdout.buffer.writer(r.encode("utf-8"))
	# sys.stdout.buffer.flush()

