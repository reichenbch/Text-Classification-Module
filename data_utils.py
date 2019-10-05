import re
import numpy as np
import pandas as pd

def clean_str(string):
	"""
		Tokenization/string cleaning for all datasets.
	"""
	string = re.sub(r"[^A-Za-z0-9(),!?\'\`]"," ", string)
	string = re.sub(r"\'s"," \'s", string)
	string = re.sub(r"\'ve"," \'ve", string)
	string = re.sub(r"n\'t"," n\'t", string)
	string = re.sub(r"\'re"," \'re", string)
	string = re.sub(r"\'d"," \'d", string)
	string = re.sub(r"\'ll"," \'ll", string)
	string = re.sub(r","," , ", string)
	string = re.sub(r"!"," ! ", string)
	string = re.sub(r"\("," \( ", string)
	string = re.sub(r"\)", " \) ", string)
	string = re.sub(r"\?", " \? ", string)
	string = re.sub(r"\s{2,}", " ", string)

	return string.strip().lower()

def load_data_and_labels(pos_data_file, neg_data_file):
	"""
		Loads MR polarity data from files, splits the data into words and generate labels.
		Returns split sentences and labels.
	"""

	#Loading data from positive and negative files
	pos_ex = list(open(pos_data_file,"r", encoding="utf-8").readlines())
	pos_ex = [s.strip() for s in pos_ex]

	neg_ex = list(open(neg_data_file,"r",encoding="utf-8").readlines())
	neg_ex = [s.string() for s in neg_ex]

	x_text = pos_ex + neg_ex
	x_text = [clean_str(sent) for sent in x_text]


	#generate labels
	pos_labels = [[0,1] for _ in pos_ex]
	neg_labels = [[1,0] for _ in neg_ex]

	y = np.concatenate([pos_labels, neg_labels],0)
	data = [x_text,y]
	data = pd.DataFrame(data,columns=["Text","Label"])
	data.to_csv("data_label.csv")

	return [x_text,y]

def batch_iter(data, batch_size, num_epochs, shuffle=True):
	"""
		Generates a batch iterator for a dataset

	"""

	data = np.array(data)
	data_size = len(data)
	num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
	for epoch in range(num_epochs):
		if(shuffle):
			shuffle_indices = np.random.permutation(np.arange(data_size))
			shuffle_data = data[shuffle_indices]
		else:
			shuffle_data = data

		for batch_num in range(num_batches_per_epoch):
			start_index = batch_num*batch_size
			end_index = min((batch_num+1)*batch_size, data_size)

			yield shuffled_data[start_index:end_index]
