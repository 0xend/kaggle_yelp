'''

Author: Eric Denovitzer
~ ./trainer.py <file_name> <limit_examples> <ratio>


'''
from ReviewTrainer import *
from BusinessTrainer import *
from UserTrainer import *
from sklearn.externals import joblib
import sys
import json
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


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
		if ex_proc % 1000 == 0: 
			logging.info('Examples processed: %d' % ex_proc)
	return results

def prepare_trainer(fname, ratio, clf, labels_ready = False):
	data = load(fname, clf.preprocess)
	if labels_ready:
		grouped_labels = None
	else:
		grouped_labels = clf.group_labels(FNAME_REVIEWS)		
	examples = clf.build_examples(data, grouped_labels)
	train = int(len(examples['labels']) * ratio)
	x = examples['feats']
	y = examples['labels']
	clf.prepare_data(x[:train], y[:train])
	clf.train()
	pred = clf.predict(x[train:])
	err = clf.get_error(pred, y[train:])
	return (clf, err)

def train_review(fname, ratio):
	rev_trainer, err = prepare_trainer(fname, ratio, ReviewTrainer(), True)
	logging.info('Review predictor error: %f' % err)
	return rev_trainer

def train_business(fname, ratio):
	biz_trainer, err = prepare_trainer(fname, ratio,BusinessTrainer())
	logging.info('Business predictor error: %f' % err)
	return biz_trainer
	
def train_user(fname, ratio):
	user_trainer, err = prepare_trainer(fname, ratio, UserTrainer())
	logging.info('User predictor error: %f' % err)
	return user_trainer
	
def experts_trainer(models, exs, labels):
	pass

def main():
	fname = sys.argv[1]
	ratio = float(sys.argv[2])

	#train_review(fname, ratio)
	#train_business(fname, ratio)
	train_user(fname, ratio)

if __name__ == '__main__':
	main()
