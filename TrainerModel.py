'''
Base class for trainers
'''

import json
import sys
from sklearn.cross_validation import KFold


class TrainerModel(object):
	def __init__(self):
		pass

	def preprocess(self, l):
		raise NotImplementedError
	
	def get_error(self, pred, y):
		dif = 0
		total = len(pred)
		for i in range(0,len(pred)):
			p = round(int(pred[i]))
			dif +=  abs(round(int(pred[i]))-int(y[i]))	
		return dif/total
	
	def _cross_validate_base(self, model, params, opt, values):
		best_score = sys.float_info.max
		best_clf = None
		kfold = KFold(n=len(self.labels), n_folds=10, indices=True)
		for v in values:
			params[opt] = v
			clf = model(**params)
			values = []
			print clf
			for train, test in kfold:
				clf.fit(self.feats[train], self.labels[train])
				pred = clf.predict(self.feats[test])
				values.append(self.get_error(pred, self.labels[test]))
			avg = sum(values)/len(values)
			if avg < best_score:
				best_clf = clf
				best_score = avg
		return (best_clf, best_score)
	
	def _cross_validate(self, **extra):
		raise NotImplementedError
	
	def group_labels(self, fname, field):
		f = open(fname)
		labels = {}
		for line in f:
			js = json.loads(line)
			try:
				labels[js[field]].append(js['votes']['useful'])
			except KeyError:
				labels[js[field]] = [js['votes']['useful']]
		return labels
	
	def build_examples(self, *args):
		raise NotImplementedError
	
	def train(self):
		raise NotImplementedError
	
	def predict(self, data):
		raise NotImplementedError
	
