'''

Author: Eric Denovitzer
~ python <file_name> <limit_examples> <ratio>


'''
from ReviewTrainer import *
from BusinessTrainer import *
import sys
import json

FNAME_REVIEWS = u'yelp_training_set_review.json'


def load(fname, prep, limit=0):
	f = open(fname)
	results = {}
	ex_proc = 0
	for line in f:
		js = json.loads(line)
		for k,v in  prep(js).items():
			results[k] = v
		ex_proc += 1
		if ex_proc % 100 == 0: 
			print 'Examples processed: %d' % ex_proc
	return results

def train_review(fname, ratio):
	rv_trainer = ReviewTrainer()
	data = load(fname, rv_trainer.preprocess)
	examples = rv_trainer.build_examples(data)
	train = int(len(examples['labels']) * ratio)
	x = examples['feats']
	y = examples['labels']
	rv_trainer.prepare_data(x[:train], y[:train])
	rv_trainer.train()
	pred = rv_trainer.predict(x[train:])
	print rv_trainer.get_error(pred, y[train:])

def train_business(fname, ratio):
	biz_trainer = BusinessTrainer()
	data = load(fname, biz_trainer.preprocess)
	grouped_labels = biz_trainer.group_labels(FNAME_REVIEWS)		
	examples = biz_trainer.build_examples(data, grouped_labels)
	train = int(len(examples['labels']) * ratio)
	x = examples['feats']
	y = examples['labels']
	biz_trainer.prepare_data(x[:train], y[:train])
	biz_trainer.train()
	pred = biz_trainer.predict(x[train:])
	print biz_trainer.get_error(pred, y[train:])

 

def main():
	fname = sys.argv[1]
	ratio = float(sys.argv[2])

	#train_review(fname, ratio)
	train_business(fname, ratio)


if __name__ == '__main__':
	main()
