import codecs

# 品詞の数
ws = 0
# 単語の品詞のリスト
wccs = {}
# 品詞リスト
wcs = {}

f = codecs.open("all-hinshi.txt", "r", "utf-8")

line = f.readline()
while line:
	l = line.strip()

	if len(l) > 0:
		# 終端文字でなかったら
		if l != "EOS":
			n = l.split("\t")

			# 単語と品詞を取得
			if len(n) == 2:
				m = ",".join(n[1].split(",")[:6])
				if m not in wcs:
					# 0は開始、1は終端文字なので、２から始める
					wcs[m] = ws + 2
					ws += 1
				wccs[n[0]] = wcs[m]

	line = f.readline()

# 単語と品詞の添え字の表を保存
r = codecs.open("all-words-parses.txt", "w", "utf-8")
for w in wccs:
	r.write(str(wccs[w]) + "," + w + "\n")
r.close()

# ファイルの元に戻る
f.seek(0)
# 単語の添え字で作られた文を保存
r = codecs.open("all-sentence-parses.txt", "w", "utf-8")

line = f.readline()
n_eos = 0
n_words = 0

while line:
	l = line.strip()
	if len(l) > 0:
		# 終端文字
		if l == "EOS":
			if n_eos == 0:
				r.write("\n")
			n_eos += 1
			n_words = 0

		# 終端文字ではない
		else:
			n = l.split("\t")
			if len(n) == 2:
				if n_words > 0:
					r.write(",")
				n_eos = 0
				n_words += 1
				m = n[1].split(",")
				if len(m) > 1:
					r.write(str(wccs[n[0]]))
	line = f.readline()

f.close()
r.close()



