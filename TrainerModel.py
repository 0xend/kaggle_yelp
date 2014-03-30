'''
Base class for trainers
'''

import json
import sys
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV

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
	
	def _cross_validate_base(self, model, grid):
		cv = KFold(n=len(self.labels), n_folds=10, indices=True)
		model = GridSearchCV(model, param_grid=grid, cv=cv)
		return model
	
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
	
