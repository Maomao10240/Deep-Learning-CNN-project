import argparse
import numpy as np
import random
import pandas as pd
import pickle


def load_vec(filename):
	w2v = {}
	with open(filename, 'r') as f:
		header = f.readline()
		vocab_size, emb_size = map(int, header.split())
		print(vocab_size)
		print(emb_size)
		for line in f:
			cline = line.split()
			w2v[cline[0]] = np.array(cline[1:], dtype=np.float64)
		return w2v, emb_size, vocab_size

def load_text(origText, subjectIDs, hospitalIDs, text_names, condition_names):
	texts = {}
	conditions = {}
	subject_ids = {}
	hadm_ids = {}
	for i in range(len(origText)):
		subject_ids[i] = origText.loc[i, subjectIDs]
		hadm_ids[i] = origText.loc[i,hospitalIDs]
		texts[i] = origText.loc[i,text_names]
		condi = []
		for c in condition_names:
			# for example: alcohol, ith patient contidion of alcohol result
			condi.append(origText.loc[i, c])
		conditions[i] = condi
	return texts, conditions, subject_ids, hadm_ids
		# if i < 3:
		# 	print(texts[i])

def main():
	global args
	parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
	parser.add_argument('--data', help="The input file", type=str, default='df_annot_text_split.pkl')
	parser.add_argument('--w2v', type =str, help ='Path to w2v.txt', default = 'w2v.txt')
	parser.add_argument('--batchsize', type =int, help ='Batch size for training', default = 1)
	parser.add_argument('--trainsize', type =int, help ='Percentage for training', default = 0.7)
	parser.add_argument('--validsize', type =int, help ='Percentage for validation', default = 0.2)
	parser.add_argument('--output_file', type =str, help ='Output of the preprocess', default = "train_val.pkl")

	args = parser.parse_args()

    #Load the word2vec file
	word2vec, emb_size, v_large = load_vec(args.w2v)
	origText = pd.read_pickle(args.data)
	subjectIDs = 'subject_id'
	hospitalIDs = 'hadm_id'
	text_names = 'merged'
	condition_names = origText.columns[2:-1]
	# load the processed origText file to dictionaries
	texts, conditions, subject_ids, hadm_ids = load_text(origText, subjectIDs, hospitalIDs, text_names,  condition_names)
	print("len of the texts:")
	print(len(texts))
	train_size = round(len(texts) * args.trainsize)
	val_size = round(len(texts) * args.validsize)
	test_size = len(texts) - train_size - val_size
	keys = list(texts.keys())
	random.shuffle(keys)

	train_keys, val_keys, test_keys = keys[:train_size], keys[train_size:train_size+val_size], keys[val_size+train_size:]
	train_texts = [texts[key] for key in train_keys]
	val_tests = [texts[key] for key in val_keys]
	test_texts = [texts[key] for key in test_keys]
	train_cond = [conditions[key] for key in train_keys]
	val_cond = [conditions[key] for key in val_keys]
	test_cond = [conditions[key] for key in test_keys]


	with open(args.output_file, "wb") as f:
		train_tar = np.array(train_cond)
		val_tar = np.array(val_cond)
		test_tar = np.array(test_cond)
		all = [train_texts, test_tar, val_tests, val_tar, test_texts, test_tar]
		pickle.dump(all,f)



	# with open(args.output_file, "rb") as f:
	# 	data = pickle.load(f)
	# print(data[:1])






if __name__ == "__main__":
    main()