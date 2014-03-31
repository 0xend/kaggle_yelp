#!/usr/bin/env python
'''

Author: Eric Denovitzer
~ ./trainer.py <file_name> <limit_examples> <ratio>


'''
from ReviewTrainer import *
from BusinessTrainer import *
from UserTrainer import *
import json
import logging
import argparse
import constants as cons
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

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
		grouped_labels = clf.group_labels(cons.FNAME_REVIEW)		
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
	

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("ratio", type=float, help="Ratio of training:test")
	parser.add_argument("--load", help="Loads models from saved files.")
	parser.add_argument("--save", help="Save trained models.")
	args = parser.parse_args()
	ratio = args.ratio
	if args.load:
		rev_trainer = joblib.load(cons.REV_MODEL)
		biz_trainer = joblib.load(cons.BIZ_MODEL)
		user_trainer = joblib.load(cons.USER_MODEL)
	else:
		rev_trainer = train_review(cons.FNAME_REVIEW, ratio)
		biz_trainer = train_business(cons.FNAME_BUSINESS, ratio)
		user_trainer = train_user(cons.FNAME_USER, ratio)

	if args.save:
		rev_trainer.save()
		biz_trainer.save()
		user_trainer.save()

if __name__ == '__main__':
	main()
