import sys
import codecs

# 現在の単語の数
wc = 0
# 単語リスト
wcc = {}

# ファイルの読み込み
f = codecs.open("alice-wakati.txt", "r", "utf-8")

line = f.readline()
while line:
	# 行の単語をリスト
	l = line.split()
	for w in l:
		if not w in wcc:
			# 0は開始文字、1は終端文字なので２から始める
			wcc[w] = wc+2
			wc += 1

	line = f.readline()

# 単語とその添え字の表を保存
r = codecs.open("all-words.txt", "w", "utf-8")
for w in wcc:
	r.write(str(wcc[w]) + "," + w + "\n")
r.close()

# 文章を単語の添え字のリストに変換する
# ファイルの初めに戻る
f.seek(0)
# 単語の添え字で表現された文章を保存する
r = codecs.open("all-sentence.txt", "w", "utf-8")

# 文章となる単語のリスト
sentence = []

# １行づつ処理
line = f.readline()
while line:
	# 行の中の単語をリストに
	l = line.split()
	for w in l:
		# 現在の単語を文に追加
		sentence.append(wcc[w])

	# 改行で保存
	for i in range(len(sentence)):
		r.write(str(sentence[i]))
		if i < len(sentence)-1:
			r.write(",")
	r.write("\n")
	sentence = []
	line = f.readline()
f.close()
r.close()

