scrape:
	python scrape_sentence.py > alice.txt

wakati:
	mecab -b 100000 -Owakati alice.txt -o alice-wakati.txt

sentence:
	python make_training_data.py

train_nsent:
	python train.py

gen_nsent:
	python generate_normal_sentence.py

get_pos:
	cat alice.txt | mecab > all-hinshi.txt

list_pw:
	python list_pos_word.py

train_po:
	python train_pos_order.py

gen_w2v:
	python generate_word2vec.py

gen_gsent:
	python generate_good_sentence.py

