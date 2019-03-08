import numpy as np
import sys
import codecs
import chainer
from model import Generate_RNN as Parse_Generate_RNN
import chainer.functions as F
from chainer import training, datasets, iterators, optimizers
from chainer.training import extensions

from gensim.models import word2vec

uses_device = -1

cp = np
if uses_device >= 0:
	import cupy as cp
	import chainer.cuda

w = codecs.open("all-words-parses.txt", "r", "utf-8")

words_parse = {}

line = w.readline()
while line:
	l = line.split(",")
	if len(l) == 2:
		r = int(l[0].strip())
		if r in words_parse:
			words_parse[r].append(l[1].strip())
		else:
			words_parse[r] = [l[1].strip()]
	line = w.readline()
w.close()

# モデルの生成
model = Parse_Generate_RNN(max(words_parse.keys())+1, 20)

# 学習済みモデルの読み込み
chainer.serializers.load_hdf5("pos_order.hdf5", model)

if uses_device >= 0:
	chainer.cuda.get_device_from_id(0).use()
	chainer.cuda.check_cuda_available()
	model.to_gpu()

words_max = 50
beam_w = 3
parses = []
model_history = [model]
cur_parses = [0]
cur_score = []
max_score = 0

def Tree_Traverse():
	global max_score

	cur_parse = cur_parses[-1]
	score = np.prod(cur_score)
	deep = len(cur_parses)

	# 枝狩り
	if max_score > 0 and deep > 5 and max_score*0.6 > score:
	 return

	# 終端文字または最大の文の長さ以上なら、文を追加して終了
	if cur_parse == 1 or deep > words_max:
		data = np.array(cur_parses)
		parses.append((score, data))
		if max_score < score:
			max_score = score
		return

	cur_model = model_history[-1].copy()
	x = cp.array([cur_parse], dtype=cp.int32)
	y = cur_model(x)
	z = F.softmax(y)
	result = z.data[0]

	if uses_device >= 0: result = chainer.cuda.to_cpu(result)
	p = np.argsort(result)[::-1]

	model_history.append(cur_model)
	for i in range(beam_w):
		cur_parses.append(p[i])
		cur_score.append(result[p[i]])
		Tree_Traverse()
		cur_parses.pop()
		cur_score.pop()
	model_history.pop()

# 木探索で探索で文章を作成
Tree_Traverse()

word_vec = word2vec.Word2Vec.load("alice-word2vec.model")

# 文章のターゲット
# target_str = ["不思議","の","国","の","アリス"]
# target_str = ["三月","うさぎ","の","お茶","会"]
target_str = ["女王"]

def similarity_word(parse, history):
	scores = []

	for i in range(len(words_parse[parse])):
		w = words_parse[parse][i]
		if w in word_vec:
			t = history[:]
			t.append(w)

			sim = word_vec.n_similarity(target_str, t)
			scores.append((sim, w))

	result = sorted(scores, key=lambda x: x[0])[::-1]
	return result[0]


result_set = sorted(parses, key=lambda x: x[0])[::-1]

for i in range(min([20, len(result_set)])):
	s, l = result_set[i]
	history = []

	for j in range(1, len(l)-1):
		score, cur_word = similarity_word(l[j], history)
		history.append(cur_word)
		sys.stdout.buffer.write(cur_word.encode("utf-8"))

	sys.stdout.buffer.write("\n".encode("utf-8"))
	sys.stdout.buffer.flush()




