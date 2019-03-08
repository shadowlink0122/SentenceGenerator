import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training


class Generate_RNN(chainer.Chain):
	def __init__(self, words_size, nodes):
		super(Generate_RNN, self).__init__()
		with self.init_scope():
			# Embed -> 埋め込み
			# LSTM -> Long Sort Term Memory
			self.embed = L.EmbedID(words_size, words_size)
			self.l1 = L.LSTM(words_size, nodes)
			self.l2 = L.LSTM(nodes, nodes)
			self.l3 = L.Linear(nodes, words_size)

	def reset_state(self):
		self.l1.reset_state()
		self.l2.reset_state()

	def __call__(self, x):
		h0 = self.embed(x)
		h1 = self.l1(h0)
		h2 = self.l2(h1)
		y = self.l3(h2)
		return y


class RNNUpdater(training.StandardUpdater):
	def __init__(self, train_iter, optimizer, device, cp):
		super(RNNUpdater, self).__init__(
			train_iter,
			optimizer,
			device=device,
		)
		self.cp = cp

	def update_core(self):
		loss = 0

		# IteratorとOptimizerの取得
		train_iter = self.get_iterator("main")
		optimizer = self.get_optimizer("main")

		# Modelの取得
		model = optimizer.target
		# 文をバッチ取得(足りないものを埋める)
		x = train_iter.__next__()
		# モデルのステータスをリセット
		model.reset_state()

		# 文の単語をRNNに学習させる
		for i in range(len(x[0])-1):
			# バッチ処理の配列
			batch = self.cp.array([s[i] for s in x], dtype=self.cp.int32)
			# 正解ラベル
			t = self.cp.array([s[i+1] for s in x], dtype=self.cp.int32)

			#終端文字なら終わり
			if self.cp.min(batch) == 1 and self.cp.max(batch) == 1:
				break

			# RNNを１回実行
			y = model(batch)
			# 損失を求める
			loss += F.softmax_cross_entropy(y, t)

		# 逆伝播
		optimizer.target.cleargrads()
		loss.backward()
		optimizer.update()




